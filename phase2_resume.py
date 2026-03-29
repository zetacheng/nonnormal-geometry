"""
Phase 2 Resume: Continue from Group C seed 123 onwards, then Group G.
Leaves ~10% GPU headroom and limits CPU to avoid system freeze.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import csv
import time
import json
import gc
from torch.amp import autocast, GradScaler

# ============================================================
# Resource Management: leave 10% GPU + reasonable CPU headroom
# ============================================================
os.environ['CUDA_MEMORY_FRACTION'] = '0.90'  # leave 10% VRAM
os.environ['OMP_NUM_THREADS'] = '4'           # limit CPU threads
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_num_threads(4)

# Limit GPU memory to 90%
torch.cuda.set_per_process_memory_fraction(0.90, 0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', 'phase2_lm')
os.makedirs(os.path.join(BASE_DIR, 'checkpoints'), exist_ok=True)


# ============================================================
# Import all model classes from phase2_lm.py
# ============================================================
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_lm import (
    TransformerLM, measure_geometry, load_data, evaluate,
    RMSNorm, RotaryEmbedding, MultiHeadAttention,
    StandardFFN, ForcedNormalFFN, AttentionAlphaBetaLM, ProjectionPathLM,
    TransformerBlock
)


# ============================================================
# Training Loop (same as original, unchanged)
# ============================================================

def train_one_config(group, seed, train_data, val_data, total_steps=20000,
                     batch_size=32, micro_batch=16, ctx_len=256, lr=3e-4, warmup_steps=1000):
    """Train with gradient accumulation: micro_batch per forward, batch_size effective."""
    accum_steps = batch_size // micro_batch  # =2 for 32/16

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    group_names = {'A': 'Standard', 'B': 'PRC(b>=0)', 'C': 'PRC(full)', 'D': 'ForcedNormal'}
    print(f"\n{'='*60}")
    print(f"Group {group}: {group_names[group]} | Seed {seed}")
    print(f"Effective batch_size={batch_size} (micro={micro_batch} x accum={accum_steps})")
    print(f"{'='*60}")

    model = TransformerLM(group=group).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = GradScaler('cuda')

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    geo_log_path = os.path.join(BASE_DIR, f'geometry_log_group{group}_seed{seed}.csv')
    metrics_log_path = os.path.join(BASE_DIR, f'metrics_log_group{group}_seed{seed}.csv')

    geo_writer = open(geo_log_path, 'w', newline='')
    geo_csv = csv.writer(geo_writer)
    geo_csv.writerow(['step', 'layer', 'henrici', 'curvature', 'excess_noise', 'frobenius', 'alpha', 'beta'])

    metrics_writer = open(metrics_log_path, 'w', newline='')
    metrics_csv = csv.writer(metrics_writer)
    metrics_csv.writerow(['step', 'seed', 'group', 'perplexity', 'loss', 'ece', 'conf_on_errors', 'high_conf_error_frac'])

    n_train = len(train_data)
    step = 0
    epoch = 0
    running_loss = 0
    t_start = time.time()

    while step < total_steps:
        epoch += 1
        perm = torch.randperm(n_train)

        # Iterate in micro_batch chunks; accumulate accum_steps before optimizer step
        data_idx = 0
        accum_count = 0
        optimizer.zero_grad()

        for i in range(0, n_train - micro_batch, micro_batch):
            if step >= total_steps:
                break

            idx = perm[i:i + micro_batch]
            x = train_data[idx].to(DEVICE)
            inp = x[:, :-1]
            tgt = x[:, 1:]

            model.train()

            with autocast('cuda'):
                logits = model(inp)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.reshape(-1))
                loss = loss / accum_steps  # scale loss for accumulation

            scaler.scale(loss).backward()
            running_loss += loss.item() * accum_steps  # un-scale for logging
            accum_count += 1

            if accum_count == accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0
                step += 1

                # Geometry logging every 500 steps
                if step % 500 == 0:
                    with torch.no_grad():
                        geo = measure_geometry(model)
                    for g in geo:
                        geo_csv.writerow([step, g['layer'], g['henrici'], g['curvature'],
                                          g['excess_noise'], g['frobenius'], g['alpha'], g['beta']])
                    geo_writer.flush()

                # Evaluation every 1000 steps
                if step % 1000 == 0:
                    avg_train_loss = running_loss / 1000
                    running_loss = 0

                    metrics = evaluate(model, val_data, max_batches=50, batch_size=micro_batch)
                    metrics_csv.writerow([step, seed, group, metrics['perplexity'],
                                          metrics['loss'], metrics['ece'],
                                          metrics['conf_on_errors'], metrics['high_conf_error_frac']])
                    metrics_writer.flush()

                    elapsed = time.time() - t_start
                    steps_per_sec = step / elapsed
                    eta = (total_steps - step) / steps_per_sec

                    last_block = model.blocks[-1]
                    ab_str = ""
                    if hasattr(last_block, 'alpha_beta'):
                        ab_str = f" | a={last_block.last_alpha:.3f} b={last_block.last_beta:.3f}"

                    print(f"  Step {step:5d}/{total_steps} | "
                          f"TrainLoss {avg_train_loss:.4f} | "
                          f"ValPPL {metrics['perplexity']:.1f} | "
                          f"ECE {metrics['ece']:.4f} | "
                          f"ConfErr {metrics['conf_on_errors']:.3f}{ab_str} | "
                          f"ETA {eta/60:.0f}m")

                # Checkpoint every 5000 steps
                if step % 5000 == 0:
                    ckpt_path = os.path.join(BASE_DIR, 'checkpoints',
                                             f'group{group}_seed{seed}_step{step}.pt')
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'step': step,
                        'group': group,
                        'seed': seed,
                    }, ckpt_path)

    geo_writer.close()
    metrics_writer.close()

    final_metrics = evaluate(model, val_data, max_batches=100, batch_size=micro_batch)
    print(f"\n  FINAL: PPL={final_metrics['perplexity']:.1f} | "
          f"ECE={final_metrics['ece']:.4f} | "
          f"ConfErr={final_metrics['conf_on_errors']:.3f}")

    ckpt_path = os.path.join(BASE_DIR, 'checkpoints',
                             f'group{group}_seed{seed}_final.pt')
    torch.save({
        'model': model.state_dict(),
        'step': step,
        'group': group,
        'seed': seed,
        'final_metrics': final_metrics,
    }, ckpt_path)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return final_metrics


# ============================================================
# Group G: Adversarial Noise Discriminator
# ============================================================

class NoiseDiscriminator(nn.Module):
    def __init__(self, d_model=384, d_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
            nn.Sigmoid()
        )
        self.gate_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, h):
        noise_score = self.net(h.detach())
        gate = 1.0 - torch.sigmoid(self.gate_scale) * noise_score
        return h * gate, noise_score


class TransformerLMWithDiscriminator(nn.Module):
    """Group A model with noise discriminator at layer 3."""
    def __init__(self, base_model, disc_layer=3):
        super().__init__()
        self.base = base_model
        self.disc_layer = disc_layer
        self.discriminator = NoiseDiscriminator(d_model=base_model.d_model)
        self.last_noise_scores = None  # for logging

        # Freeze base model
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, idx):
        B, T = idx.shape
        x = self.base.tok_embed(idx) * math.sqrt(self.base.d_model)
        x = self.base.drop(x)

        for i, block in enumerate(self.base.blocks):
            # Attention
            x = x + block.attn(block.norm1(x), mask=self.base.mask)

            # FFN
            normed = block.norm2(x)
            ffn_out = block.ffn(normed)

            if i == self.disc_layer:
                ffn_out, noise_score = self.discriminator(ffn_out)
                self.last_noise_scores = noise_score

            x = x + ffn_out

        x = self.base.norm(x)
        return self.base.head(x)


def train_group_g(seed, train_data, val_data, total_steps=5000,
                  warmup_steps=1000, batch_size=32):
    """Train Group G: discriminator on frozen Group A model."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*60}")
    print(f"Group G: Adversarial Noise Discriminator | Seed {seed}")
    print(f"{'='*60}")

    # Load best Group A checkpoint
    best_ppl = float('inf')
    best_ckpt = None
    for s in [42, 123, 456]:
        ckpt_path = os.path.join(BASE_DIR, 'checkpoints', f'groupA_seed{s}_final.pt')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            ppl = ckpt.get('final_metrics', {}).get('perplexity', float('inf'))
            if ppl < best_ppl:
                best_ppl = ppl
                best_ckpt = ckpt_path

    print(f"Loading base model from: {os.path.basename(best_ckpt)} (PPL={best_ppl:.1f})")

    base_model = TransformerLM(group='A')
    ckpt = torch.load(best_ckpt, map_location='cpu', weights_only=False)
    base_model.load_state_dict(ckpt['model'])

    model = TransformerLMWithDiscriminator(base_model, disc_layer=3).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,} | Trainable: {trainable:,} ({100*trainable/total_params:.2f}%)")

    # Separate optimizers for discriminator net and gate_scale
    optimizer_disc = torch.optim.Adam(model.discriminator.net.parameters(), lr=1e-4)
    optimizer_gate = torch.optim.Adam([model.discriminator.gate_scale], lr=1e-3)
    scaler = GradScaler('cuda')

    # Logging
    metrics_log_path = os.path.join(BASE_DIR, f'metrics_log_groupG_seed{seed}.csv')
    metrics_writer = open(metrics_log_path, 'w', newline='')
    metrics_csv = csv.writer(metrics_writer)
    metrics_csv.writerow(['step', 'seed', 'group', 'perplexity', 'loss', 'ece',
                          'conf_on_errors', 'high_conf_error_frac',
                          'disc_accuracy', 'mean_ns_correct', 'mean_ns_incorrect', 'gate_scale'])

    # Also log geometry for consistency
    geo_log_path = os.path.join(BASE_DIR, f'geometry_log_groupG_seed{seed}.csv')
    geo_writer = open(geo_log_path, 'w', newline='')
    geo_csv = csv.writer(geo_writer)
    geo_csv.writerow(['step', 'layer', 'henrici', 'curvature', 'excess_noise', 'frobenius', 'alpha', 'beta'])

    n_train = len(train_data)
    step = 0
    epoch = 0
    running_loss = 0
    t_start = time.time()

    while step < total_steps:
        epoch += 1
        perm = torch.randperm(n_train)

        for i in range(0, n_train - batch_size, batch_size):
            if step >= total_steps:
                break

            idx_batch = perm[i:i + batch_size]
            x = train_data[idx_batch].to(DEVICE)
            inp = x[:, :-1]
            tgt = x[:, 1:]

            step += 1

            # ---- Phase 1: Warmup (steps 1-1000): train discriminator only ----
            if step <= warmup_steps:
                model.eval()
                with torch.no_grad():
                    with autocast('cuda'):
                        logits = model(inp)
                    logits_f = logits.float()
                    pred = logits_f.argmax(dim=-1)
                    correct = (pred == tgt).float()  # (B, T)

                # Train discriminator
                model.discriminator.train()
                noise_score = model.last_noise_scores  # (B, T, d_model)
                if noise_score is not None:
                    # For correct predictions: noise_score should be low
                    # For incorrect predictions: noise_score should be high
                    correct_expanded = correct.unsqueeze(-1)  # (B, T, 1)
                    loss_disc = torch.mean(correct_expanded * noise_score ** 2 +
                                          (1 - correct_expanded) * (1 - noise_score) ** 2)
                    # Sparsity regularization
                    loss_disc = loss_disc + 0.01 * torch.mean(noise_score)

                    optimizer_disc.zero_grad()
                    loss_disc.backward()
                    optimizer_disc.step()

                    running_loss += loss_disc.item()

            # ---- Phase 2 (steps 1001+): Joint training with gating ----
            else:
                within_cycle = (step - warmup_steps) % 50

                if within_cycle < 40:
                    # Train gate_scale via main CE loss
                    model.discriminator.gate_scale.requires_grad = True
                    for p in model.discriminator.net.parameters():
                        p.requires_grad = False

                    model.eval()
                    model.discriminator.train()

                    with autocast('cuda'):
                        logits = model(inp)
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.reshape(-1))

                    optimizer_gate.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer_gate)
                    scaler.step(optimizer_gate)
                    scaler.update()

                    running_loss += loss.item()

                else:
                    # Update discriminator
                    model.discriminator.gate_scale.requires_grad = False
                    for p in model.discriminator.net.parameters():
                        p.requires_grad = True

                    model.eval()
                    with torch.no_grad():
                        with autocast('cuda'):
                            logits = model(inp)
                        logits_f = logits.float()
                        pred = logits_f.argmax(dim=-1)
                        correct = (pred == tgt).float()

                    model.discriminator.train()
                    noise_score = model.last_noise_scores
                    if noise_score is not None:
                        correct_expanded = correct.unsqueeze(-1)
                        loss_disc = torch.mean(correct_expanded * noise_score ** 2 +
                                              (1 - correct_expanded) * (1 - noise_score) ** 2)
                        loss_disc = loss_disc + 0.01 * torch.mean(noise_score)

                        optimizer_disc.zero_grad()
                        loss_disc.backward()
                        optimizer_disc.step()

                        running_loss += loss_disc.item()

            # Evaluation every 500 steps
            if step % 500 == 0:
                avg_loss = running_loss / 500
                running_loss = 0

                # Full eval
                metrics = evaluate_group_g(model, val_data, batch_size=batch_size)

                gate_val = torch.sigmoid(model.discriminator.gate_scale).item()
                metrics_csv.writerow([step, seed, 'G', metrics['perplexity'],
                                      metrics['loss'], metrics['ece'],
                                      metrics['conf_on_errors'], metrics['high_conf_error_frac'],
                                      metrics['disc_accuracy'],
                                      metrics['mean_ns_correct'], metrics['mean_ns_incorrect'],
                                      gate_val])
                metrics_writer.flush()

                elapsed = time.time() - t_start
                steps_per_sec = step / elapsed
                eta = (total_steps - step) / steps_per_sec

                print(f"  Step {step:5d}/{total_steps} | "
                      f"PPL {metrics['perplexity']:.1f} | "
                      f"ConfErr {metrics['conf_on_errors']:.4f} | "
                      f"DiscAcc {metrics['disc_accuracy']:.3f} | "
                      f"Gate {gate_val:.3f} | "
                      f"NS_cor {metrics['mean_ns_correct']:.3f} NS_inc {metrics['mean_ns_incorrect']:.3f} | "
                      f"ETA {eta/60:.0f}m")

            # Geometry logging every 500 steps (base model weights don't change but disc does)
            if step % 500 == 0:
                with torch.no_grad():
                    geo = measure_geometry(model.base)
                for g in geo:
                    geo_csv.writerow([step, g['layer'], g['henrici'], g['curvature'],
                                      g['excess_noise'], g['frobenius'], 1.0, 0.0])
                geo_writer.flush()

            # Checkpoint every 1000 steps
            if step % 1000 == 0:
                ckpt_path = os.path.join(BASE_DIR, 'checkpoints',
                                         f'groupG_seed{seed}_step{step}.pt')
                torch.save({
                    'discriminator': model.discriminator.state_dict(),
                    'step': step,
                    'seed': seed,
                }, ckpt_path)

    geo_writer.close()
    metrics_writer.close()

    # Final evaluation
    final_metrics = evaluate_group_g(model, val_data, max_batches=100, batch_size=batch_size)
    print(f"\n  FINAL: PPL={final_metrics['perplexity']:.1f} | "
          f"ECE={final_metrics['ece']:.4f} | "
          f"ConfErr={final_metrics['conf_on_errors']:.4f} | "
          f"DiscAcc={final_metrics['disc_accuracy']:.3f} | "
          f"Gate={torch.sigmoid(model.discriminator.gate_scale).item():.3f}")

    ckpt_path = os.path.join(BASE_DIR, 'checkpoints', f'groupG_seed{seed}_final.pt')
    torch.save({
        'discriminator': model.discriminator.state_dict(),
        'step': step,
        'seed': seed,
        'final_metrics': final_metrics,
    }, ckpt_path)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return final_metrics


@torch.no_grad()
def evaluate_group_g(model, val_data, max_batches=50, batch_size=32):
    """Evaluate Group G model with discriminator-specific metrics."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    all_wrong_confs = []
    all_bins = np.zeros(10)
    all_bin_correct = np.zeros(10)
    all_bin_conf = np.zeros(10)

    # Discriminator-specific
    disc_correct_count = 0
    disc_total_count = 0
    ns_on_correct = []
    ns_on_incorrect = []

    n_batches = min(max_batches, len(val_data) // batch_size)

    for i in range(n_batches):
        x = val_data[i * batch_size:(i + 1) * batch_size].to(DEVICE)
        inp = x[:, :-1]
        tgt = x[:, 1:]

        with autocast('cuda'):
            logits = model(inp)
        logits_f = logits.float()
        loss = F.cross_entropy(logits_f.view(-1, logits_f.size(-1)), tgt.reshape(-1))
        total_loss += loss.item() * tgt.numel()
        total_tokens += tgt.numel()

        probs = F.softmax(logits_f, dim=-1)
        conf, pred = probs.max(dim=-1)
        wrong = (pred != tgt)
        correct = ~wrong

        if wrong.any():
            all_wrong_confs.extend(conf[wrong].cpu().numpy())

        # ECE bins
        conf_flat = conf.cpu().numpy().ravel()
        correct_flat = correct.cpu().numpy().ravel().astype(float)
        for b in range(10):
            lo, hi = b / 10, (b + 1) / 10
            mask = (conf_flat > lo) & (conf_flat <= hi)
            all_bins[b] += mask.sum()
            all_bin_correct[b] += correct_flat[mask].sum()
            all_bin_conf[b] += conf_flat[mask].sum()

        # Discriminator metrics
        noise_score = model.last_noise_scores
        if noise_score is not None:
            mean_ns = noise_score.mean(dim=-1)  # (B, T)
            if correct.any():
                ns_on_correct.extend(mean_ns[correct].cpu().numpy())
            if wrong.any():
                ns_on_incorrect.extend(mean_ns[wrong].cpu().numpy())

            # Disc accuracy: does higher noise_score correspond to incorrect?
            threshold = 0.5
            disc_pred_wrong = (mean_ns > threshold)
            disc_correct_count += ((disc_pred_wrong == wrong).float().sum()).item()
            disc_total_count += wrong.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))

    ece = 0
    for b in range(10):
        if all_bins[b] > 0:
            acc_b = all_bin_correct[b] / all_bins[b]
            conf_b = all_bin_conf[b] / all_bins[b]
            ece += all_bins[b] * abs(acc_b - conf_b)
    ece /= max(all_bins.sum(), 1)

    wrong_confs = np.array(all_wrong_confs) if all_wrong_confs else np.array([0])
    conf_on_errors = wrong_confs.mean()
    high_conf_frac = (wrong_confs > 0.8).mean()

    disc_accuracy = disc_correct_count / max(disc_total_count, 1)
    mean_ns_correct = np.mean(ns_on_correct) if ns_on_correct else 0.0
    mean_ns_incorrect = np.mean(ns_on_incorrect) if ns_on_incorrect else 0.0

    return {
        'loss': avg_loss,
        'perplexity': ppl,
        'ece': ece,
        'conf_on_errors': conf_on_errors,
        'high_conf_error_frac': high_conf_frac,
        'disc_accuracy': disc_accuracy,
        'mean_ns_correct': mean_ns_correct,
        'mean_ns_incorrect': mean_ns_incorrect,
    }


# ============================================================
# Main: Resume from C seed 123, then D, then G
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 2 RESUME: C(123,456) -> D(all) -> G(all)")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print(f"GPU memory fraction: 90% (10% reserved for system)")
    print(f"CPU threads: {torch.get_num_threads()}")
    print("=" * 60)

    train_data, val_data = load_data(ctx_len=256)

    all_results = {}
    total_steps = 20000

    # ---- Resume: Group C seeds 123, 456 ----
    for seed in [123, 456]:
        key = f"C_{seed}"
        print(f"\n>>> Starting {key} (fresh, no checkpoint to resume from)")
        result = train_one_config(
            group='C', seed=seed,
            train_data=train_data, val_data=val_data,
            total_steps=total_steps,
        )
        all_results[key] = result

    # ---- Group D: all 3 seeds ----
    for seed in [42, 123, 456]:
        key = f"D_{seed}"
        result = train_one_config(
            group='D', seed=seed,
            train_data=train_data, val_data=val_data,
            total_steps=total_steps,
        )
        all_results[key] = result

    # ---- Group G: all 3 seeds ----
    print("\n" + "=" * 60)
    print("Starting Group G: Adversarial Noise Discriminator")
    print("=" * 60)

    for seed in [42, 123, 456]:
        key = f"G_{seed}"
        result = train_group_g(
            seed=seed,
            train_data=train_data, val_data=val_data,
            total_steps=5000,
            warmup_steps=1000,
        )
        all_results[key] = result

    # ---- Final Summary ----
    print("\n" + "=" * 60)
    print("PHASE 2 RESUME COMPLETE")
    print("=" * 60)

    # Save results
    summary_path = os.path.join(BASE_DIR, 'results_summary_resume.json')
    with open(summary_path, 'w') as f:
        json.dump({k: v for k, v in all_results.items()}, f, indent=2, default=str)

    # Print summary
    for group_key in ['C', 'D', 'G']:
        seeds_done = [k for k in all_results if k.startswith(group_key + '_')]
        if seeds_done:
            ppls = [all_results[k]['perplexity'] for k in seeds_done]
            confs = [all_results[k]['conf_on_errors'] for k in seeds_done]
            print(f"  Group {group_key}: "
                  f"PPL={np.mean(ppls):.1f}+/-{np.std(ppls):.1f} | "
                  f"ConfErr={np.mean(confs):.4f}+/-{np.std(confs):.4f}")


if __name__ == '__main__':
    main()
