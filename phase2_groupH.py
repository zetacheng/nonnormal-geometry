"""
Phase 2 Group H: Training-Time Noise Discriminator
Unlike Group G (post-hoc on frozen model), Group H integrates the discriminator
DURING training. The model learns to work WITH noise suppression from the start.

Key differences from G:
  - Base model is NOT frozen — trains from scratch like Group A
  - Discriminator trains alongside the model
  - Gate suppression is active during training, so model adapts to it
  - Phase 1 (warmup): train model normally, warm up discriminator (no gating)
  - Phase 2 (joint): model + discriminator + gate all train together
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

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # required to prevent async CUDA hang
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(0.90, 0)

DEVICE = torch.device('cuda')
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', 'phase2_lm')
os.makedirs(os.path.join(BASE_DIR, 'checkpoints'), exist_ok=True)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phase2_lm import (
    TransformerLM, measure_geometry, load_data, evaluate,
    RMSNorm, RotaryEmbedding, MultiHeadAttention,
    StandardFFN, TransformerBlock
)


# ============================================================
# Noise Discriminator (same architecture as Group G)
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

    def forward(self, h, apply_gate=True):
        noise_score = self.net(h.detach())  # detach: disc doesn't backprop into base
        if apply_gate:
            gate = 1.0 - torch.sigmoid(self.gate_scale) * noise_score
            return h * gate, noise_score
        else:
            return h, noise_score


# ============================================================
# Group H Model: Trainable base + discriminator
# ============================================================

class GroupHModel(nn.Module):
    def __init__(self, d_model=384, n_heads=6, d_ff=1536, n_layers=6,
                 max_seq=256, dropout=0.1, disc_layer=3):
        super().__init__()
        self.d_model = d_model
        self.disc_layer = disc_layer

        # Standard transformer (same as Group A)
        self.tok_embed = nn.Embedding(50257, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, group='A')
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(50257, d_model, bias=False)  # will be tied
        self.head = nn.Linear(d_model, 50257, bias=False)
        self.head.weight = self.tok_embed.weight  # weight tying

        # Causal mask
        mask = torch.tril(torch.ones(max_seq, max_seq))
        self.register_buffer('mask', mask)

        # Discriminator at disc_layer
        self.discriminator = NoiseDiscriminator(d_model=d_model)

        self.last_noise_scores = None
        self.last_ffn_out = None  # saved for disc update
        self.gating_active = False  # toggled externally

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, idx, return_noise_score=False):
        B, T = idx.shape
        x = self.tok_embed(idx) * math.sqrt(self.d_model)
        x = self.drop(x)

        noise_score_out = None
        for i, block in enumerate(self.blocks):
            # Attention
            x = x + block.attn(block.norm1(x), mask=self.mask)
            # FFN
            normed = block.norm2(x)
            ffn_out = block.ffn(normed)

            if i == self.disc_layer:
                self.last_ffn_out = ffn_out.detach()  # save for disc update
                ffn_out, noise_score_out = self.discriminator(
                    ffn_out, apply_gate=self.gating_active
                )
                self.last_noise_scores = noise_score_out

            x = x + ffn_out

        x = self.norm(x)
        logits = self.head(x)

        if return_noise_score:
            return logits, noise_score_out
        return logits

    def base_parameters(self):
        """All parameters except discriminator."""
        disc_params = set(id(p) for p in self.discriminator.parameters())
        return [p for p in self.parameters() if id(p) not in disc_params]

    def get_ffn_weights(self):
        """For geometry measurement compatibility."""
        weights = []
        for block in self.blocks:
            if hasattr(block.ffn, 'w1'):
                weights.append(block.ffn.w1.weight)
        return weights


# ============================================================
# Evaluation with discriminator metrics
# ============================================================

@torch.no_grad()
def evaluate_groupH(model, val_data, max_batches=50, batch_size=16):
    model.eval()
    total_loss = 0
    total_tokens = 0
    all_wrong_confs = []
    all_bins = np.zeros(10)
    all_bin_correct = np.zeros(10)
    all_bin_conf = np.zeros(10)
    ns_on_correct = []
    ns_on_incorrect = []
    disc_correct_count = 0
    disc_total_count = 0

    n_batches = min(max_batches, len(val_data) // batch_size)

    for i in range(n_batches):
        x = val_data[i * batch_size:(i + 1) * batch_size].to(DEVICE)
        inp = x[:, :-1]
        tgt = x[:, 1:]

        with autocast('cuda'):
            logits, noise_score = model(inp, return_noise_score=True)
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

        conf_flat = conf.cpu().numpy().ravel()
        correct_flat = correct.cpu().numpy().ravel().astype(float)
        for b in range(10):
            lo, hi = b / 10, (b + 1) / 10
            m = (conf_flat > lo) & (conf_flat <= hi)
            all_bins[b] += m.sum()
            all_bin_correct[b] += correct_flat[m].sum()
            all_bin_conf[b] += conf_flat[m].sum()

        if noise_score is not None:
            mean_ns = noise_score.mean(dim=-1)
            if correct.any():
                ns_on_correct.extend(mean_ns[correct].cpu().numpy())
            if wrong.any():
                ns_on_incorrect.extend(mean_ns[wrong].cpu().numpy())
            threshold = 0.5
            disc_pred_wrong = (mean_ns > threshold)
            disc_correct_count += (disc_pred_wrong == wrong).sum().item()
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

    return {
        'loss': avg_loss,
        'perplexity': ppl,
        'ece': ece,
        'conf_on_errors': wrong_confs.mean(),
        'high_conf_error_frac': (wrong_confs > 0.8).mean(),
        'disc_accuracy': disc_correct_count / max(disc_total_count, 1),
        'mean_ns_correct': np.mean(ns_on_correct) if ns_on_correct else 0.0,
        'mean_ns_incorrect': np.mean(ns_on_incorrect) if ns_on_incorrect else 0.0,
        'gate_scale': torch.sigmoid(model.discriminator.gate_scale).item(),
    }


# ============================================================
# Geometry measurement (reuse from phase2_lm but for GroupH)
# ============================================================

def measure_geometry_H(model):
    results = []
    for i, block in enumerate(model.blocks):
        if hasattr(block.ffn, 'w1') and hasattr(block.ffn, 'w2'):
            W1 = block.ffn.w1.weight.detach().float().cpu()
            W2 = block.ffn.w2.weight.detach().float().cpu()
            M = W2 @ W1
        else:
            continue

        fro_sq = torch.sum(M ** 2).item()
        try:
            eigs = torch.linalg.eigvals(M)
            spec_sq = torch.sum(torch.abs(eigs) ** 2).item()
            henrici = np.sqrt(max(fro_sq - spec_sq, 0)) / np.sqrt(max(fro_sq, 1e-12))
            H = (M + M.T) / 2
            K = (M - M.T) / 2
            curvature = torch.norm(H @ K - K @ H, p='fro').item()
            excess_noise = max(fro_sq - spec_sq, 0)
            frobenius = torch.norm(M, p='fro').item()
        except Exception:
            henrici, curvature, excess_noise, frobenius = 0, 0, 0, 0

        results.append({
            'layer': i, 'henrici': henrici, 'curvature': curvature,
            'excess_noise': excess_noise, 'frobenius': frobenius,
            'alpha': 1.0, 'beta': 0.0,
        })
    return results


# ============================================================
# Training Loop
# ============================================================

def train_groupH(seed, train_data, val_data, total_steps=20000,
                 warmup_disc_steps=2000, micro_batch=16, accum_steps=2,
                 lr=3e-4, warmup_lr_steps=1000, disc_layer=3):

    effective_batch = micro_batch * accum_steps
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    print(f"\n{'='*60}")
    print(f"Group H: Training-Time Noise Discriminator | Seed {seed}")
    print(f"Effective batch_size={effective_batch} (micro={micro_batch} x accum={accum_steps})")
    print(f"Disc warmup: {warmup_disc_steps} steps, then joint training")
    print(f"Discriminator at layer {disc_layer}")
    print(f"{'='*60}")

    model = GroupHModel(disc_layer=disc_layer).to(DEVICE)

    base_count = sum(p.numel() for p in model.base_parameters())
    disc_count = sum(p.numel() for p in model.discriminator.parameters())
    print(f"Base params: {base_count:,} | Disc params: {disc_count:,} ({disc_count/base_count*100:.1f}%)")

    # Separate optimizers
    optimizer_base = torch.optim.AdamW(model.base_parameters(), lr=lr,
                                        betas=(0.9, 0.95), weight_decay=0.1)
    optimizer_disc = torch.optim.Adam(list(model.discriminator.net.parameters()), lr=1e-4)
    optimizer_gate = torch.optim.Adam([model.discriminator.gate_scale], lr=5e-3)
    scaler = GradScaler('cuda')

    # LR schedule for base model
    def lr_schedule(step):
        if step < warmup_lr_steps:
            return step / warmup_lr_steps
        progress = (step - warmup_lr_steps) / (total_steps - warmup_lr_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer_base, lr_schedule)

    # Logging
    metrics_path = os.path.join(BASE_DIR, f'metrics_log_groupH_seed{seed}.csv')
    geo_path = os.path.join(BASE_DIR, f'geometry_log_groupH_seed{seed}.csv')

    mf = open(metrics_path, 'w', newline='', buffering=1)
    mc = csv.writer(mf)
    mc.writerow(['step', 'seed', 'group', 'perplexity', 'loss', 'ece',
                 'conf_on_errors', 'high_conf_error_frac',
                 'disc_accuracy', 'mean_ns_correct', 'mean_ns_incorrect', 'gate_scale'])
    mf.flush()

    gf = open(geo_path, 'w', newline='', buffering=1)
    gc_w = csv.writer(gf)
    gc_w.writerow(['step', 'layer', 'henrici', 'curvature', 'excess_noise', 'frobenius', 'alpha', 'beta'])
    gf.flush()

    n_train = len(train_data)
    step = 0
    micro_count = 0
    running_loss = 0
    t_start = time.time()

    print(f"\n  Phase 1: Base model warmup + disc training (steps 1-{warmup_disc_steps})")
    print(f"  (gating OFF, model trains normally, disc learns to detect errors)")

    while step < total_steps:
        perm = torch.randperm(n_train)

        for i in range(0, n_train - micro_batch, micro_batch):
            if step >= total_steps:
                break

            idx = perm[i:i + micro_batch]
            x = train_data[idx].to(DEVICE)
            inp = x[:, :-1]
            tgt = x[:, 1:]

            micro_count += 1
            is_boundary = (micro_count % accum_steps == 0)

            # ============================================
            # Phase 1: Normal training + disc warmup (no gating)
            # ============================================
            if step < warmup_disc_steps:
                model.gating_active = False
                model.train()

                # --- Train base model (CE loss) ---
                with autocast('cuda'):
                    logits = model(inp)
                loss_ce = F.cross_entropy(logits.float().view(-1, 50257), tgt.reshape(-1))
                loss_scaled = loss_ce / accum_steps
                scaler.scale(loss_scaled).backward()
                running_loss += loss_ce.item()

                # Save noise_scores and predictions from this micro-batch for disc update
                with torch.no_grad():
                    last_pred = logits.float().argmax(dim=-1).detach()
                    last_correct = (last_pred == tgt).float()
                    last_ns = model.last_noise_scores  # saved during forward

                if is_boundary:
                    scaler.unscale_(optimizer_base)
                    torch.nn.utils.clip_grad_norm_(model.base_parameters(), 1.0)
                    scaler.step(optimizer_base)
                    scaler.update()
                    optimizer_base.zero_grad()
                    scheduler.step()

                    # --- Train discriminator using cached ffn_out ---
                    # No extra forward pass needed!
                    cached_ffn = model.last_ffn_out
                    if cached_ffn is not None:
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        model.discriminator.train()
                        # Re-run just the disc net on cached FFN output
                        ns_fresh = model.discriminator.net(cached_ffn.detach().float())
                        correct_exp = last_correct.unsqueeze(-1)
                        loss_disc = torch.mean(correct_exp * ns_fresh ** 2 +
                                              (1 - correct_exp) * (1 - ns_fresh) ** 2)
                        loss_disc = loss_disc + 0.01 * torch.mean(ns_fresh)
                        optimizer_disc.zero_grad()
                        loss_disc.backward()
                        optimizer_disc.step()

                    step += 1

            # ============================================
            # Phase 2: Joint training with gating active
            # ============================================
            else:
                model.gating_active = True
                cycle_pos = (step - warmup_disc_steps) % 20

                if cycle_pos < 15:
                    # --- Train base model + gate with CE loss only ---
                    # Gate learns via CE: if suppression helps, CE drops
                    model.train()
                    model.discriminator.gate_scale.requires_grad_(True)

                    with autocast('cuda'):
                        logits = model(inp)
                    loss_ce = F.cross_entropy(logits.float().view(-1, 50257), tgt.reshape(-1))

                    # Add small conf penalty without holding full softmax in graph
                    # Use CE loss directly — gate_scale gets gradient through gated FFN
                    joint_step = max(step - warmup_disc_steps, 0)
                    ramp = min(1.0, joint_step / 5000)
                    # Entropy regularization: encourage model to be less confident overall
                    # This is cheaper than softmax -> conf -> wrong mask
                    log_probs = F.log_softmax(logits.float(), dim=-1)
                    entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()
                    # Negative entropy = encourage higher entropy = less overconfident
                    loss = loss_ce - 0.1 * ramp * entropy

                    loss_scaled = loss / accum_steps
                    scaler.scale(loss_scaled).backward()
                    running_loss += loss_ce.item()

                    if is_boundary:
                        scaler.unscale_(optimizer_base)
                        torch.nn.utils.clip_grad_norm_(model.base_parameters(), 1.0)
                        scaler.step(optimizer_base)
                        scaler.update()
                        optimizer_base.zero_grad()
                        optimizer_gate.step()
                        optimizer_gate.zero_grad()
                        scheduler.step()
                        step += 1

                else:
                    # --- Update discriminator using cached ffn_out ---
                    # No extra forward pass! Use cached data from main forward
                    model.train()

                    with autocast('cuda'):
                        logits = model(inp)
                    loss_ce = F.cross_entropy(logits.float().view(-1, 50257), tgt.reshape(-1))
                    loss_scaled = loss_ce / accum_steps
                    scaler.scale(loss_scaled).backward()
                    running_loss += loss_ce.item()

                    # Save predictions for disc
                    with torch.no_grad():
                        pred = logits.float().argmax(dim=-1)
                        correct = (pred == tgt).float()

                    cached_ffn = model.last_ffn_out
                    if cached_ffn is not None:
                        torch.cuda.empty_cache()
                        model.discriminator.train()
                        model.discriminator.gate_scale.requires_grad_(False)
                        ns = model.discriminator.net(cached_ffn.detach().float())
                        correct_exp = correct.unsqueeze(-1)
                        loss_disc = torch.mean(correct_exp * ns ** 2 +
                                              (1 - correct_exp) * (1 - ns) ** 2)
                        loss_disc = loss_disc + 0.01 * torch.mean(ns)
                        optimizer_disc.zero_grad()
                        loss_disc.backward()
                        optimizer_disc.step()
                        model.discriminator.gate_scale.requires_grad_(True)

                    if is_boundary:
                        scaler.unscale_(optimizer_base)
                        torch.nn.utils.clip_grad_norm_(model.base_parameters(), 1.0)
                        scaler.step(optimizer_base)
                        scaler.update()
                        optimizer_base.zero_grad()
                        scheduler.step()
                        step += 1

            if not is_boundary:
                continue

            if step == warmup_disc_steps + 1:
                print(f"\n  Phase 2: Joint training with gating (steps {warmup_disc_steps+1}-{total_steps})")

            # ============================================
            # Geometry logging every 500 steps
            # ============================================
            if step % 500 == 0 and step > 0:
                torch.cuda.synchronize()
                with torch.no_grad():
                    geo = measure_geometry_H(model)
                for g in geo:
                    gc_w.writerow([step, g['layer'], g['henrici'], g['curvature'],
                                   g['excess_noise'], g['frobenius'], g['alpha'], g['beta']])
                gf.flush()

            # ============================================
            # Evaluation every 1000 steps
            # ============================================
            if step % 1000 == 0 and step > 0:
                avg_loss = running_loss / 1000
                running_loss = 0

                model.gating_active = True  # eval with gating
                metrics = evaluate_groupH(model, val_data, batch_size=micro_batch)

                gate_val = torch.sigmoid(model.discriminator.gate_scale).item()
                mc.writerow([step, seed, 'H', metrics['perplexity'], metrics['loss'],
                             metrics['ece'], metrics['conf_on_errors'],
                             metrics['high_conf_error_frac'], metrics['disc_accuracy'],
                             metrics['mean_ns_correct'], metrics['mean_ns_incorrect'],
                             gate_val])
                mf.flush()

                elapsed = time.time() - t_start
                eta = (total_steps - step) / (step / elapsed)
                phase = "Warmup" if step <= warmup_disc_steps else "Joint"

                print(f"  Step {step:5d}/{total_steps} [{phase}] | "
                      f"PPL {metrics['perplexity']:.1f} | "
                      f"ECE {metrics['ece']:.4f} | "
                      f"ConfErr {metrics['conf_on_errors']:.4f} | "
                      f"DiscAcc {metrics['disc_accuracy']:.3f} | "
                      f"NS {metrics['mean_ns_correct']:.3f}/{metrics['mean_ns_incorrect']:.3f} | "
                      f"Gate {gate_val:.4f} | "
                      f"ETA {eta/60:.0f}m", flush=True)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            # Checkpoint every 5000 steps
            if step % 5000 == 0 and step > 0:
                ckpt_path = os.path.join(BASE_DIR, 'checkpoints',
                                         f'groupH_seed{seed}_step{step}.pt')
                torch.save({
                    'model': model.state_dict(),
                    'optimizer_base': optimizer_base.state_dict(),
                    'step': step, 'seed': seed,
                }, ckpt_path)

    mf.close()
    gf.close()

    # Final eval
    model.gating_active = True
    final = evaluate_groupH(model, val_data, max_batches=100, batch_size=micro_batch)
    gate_val = torch.sigmoid(model.discriminator.gate_scale).item()

    print(f"\n  FINAL: PPL={final['perplexity']:.1f} | "
          f"ECE={final['ece']:.4f} | "
          f"ConfErr={final['conf_on_errors']:.4f} | "
          f"DiscAcc={final['disc_accuracy']:.3f} | "
          f"Gate={gate_val:.4f}")

    ckpt_path = os.path.join(BASE_DIR, 'checkpoints', f'groupH_seed{seed}_final.pt')
    torch.save({
        'model': model.state_dict(),
        'step': step, 'seed': seed,
        'final_metrics': final,
    }, ckpt_path)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return final


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("GROUP H: Training-Time Noise Discriminator")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print("=" * 60)

    train_data, val_data = load_data(ctx_len=256)

    all_results = {}
    seeds = [42, 123, 456]
    # Check for completed seeds to allow resuming
    for seed in seeds:
        summary_check = os.path.join(BASE_DIR, 'checkpoints', f'groupH_seed{seed}_final.pt')
        if os.path.exists(summary_check):
            ckpt = torch.load(summary_check, map_location='cpu', weights_only=False)
            all_results[f"H_{seed}"] = ckpt.get('final_metrics', {})
            print(f"  Seed {seed} already complete, skipping.")
            continue
        key = f"H_{seed}"
        result = train_groupH(seed=seed, train_data=train_data, val_data=val_data,
                              total_steps=20000, warmup_disc_steps=2000,
                              micro_batch=4, accum_steps=8)
        all_results[key] = result

    print("\n" + "=" * 60)
    print("GROUP H COMPLETE")
    print("=" * 60)

    ppls = [all_results[k]['perplexity'] for k in all_results]
    confs = [all_results[k]['conf_on_errors'] for k in all_results]
    print(f"  PPL: {np.mean(ppls):.1f} ± {np.std(ppls):.1f}")
    print(f"  ConfOnErr: {np.mean(confs):.4f} ± {np.std(confs):.4f}")

    summary_path = os.path.join(BASE_DIR, 'groupH_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
