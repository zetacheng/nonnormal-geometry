"""
Phase 2 Group G: Adversarial Noise Discriminator
Uses Group A's frozen checkpoint. Only trains discriminator + gate (~70K params).
Run after Groups A-D complete.
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
from torch.amp import autocast, GradScaler

# Import model and data loading from phase2_lm
from phase2_lm import TransformerLM, load_data, RMSNorm

# Resource management: leave 10% GPU + limit CPU
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(0.90, 0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', 'phase2_lm')
os.makedirs(os.path.join(BASE_DIR, 'checkpoints'), exist_ok=True)


# ============================================================
# Noise Discriminator
# ============================================================

class NoiseDiscriminator(nn.Module):
    """Per-dimension noise scorer. Learns which hidden dimensions
    are associated with incorrect predictions."""
    def __init__(self, d_model=384, d_hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_model),
            nn.Sigmoid()  # per-dimension noise score in [0, 1]
        )
        self.gate_scale = nn.Parameter(torch.tensor(0.0))

    def forward(self, h):
        noise_score = self.net(h.detach())  # detach: no gradient into base model
        gate = 1.0 - torch.sigmoid(self.gate_scale) * noise_score
        return h * gate, noise_score


# ============================================================
# Group G Model Wrapper
# ============================================================

class GroupGModel(nn.Module):
    """Wraps a frozen Group A model with a noise discriminator at a target layer."""
    def __init__(self, base_model, disc_layer=3, d_model=384):
        super().__init__()
        self.base = base_model
        self.disc_layer = disc_layer
        self.discriminator = NoiseDiscriminator(d_model=d_model)

        # Freeze base model
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, idx, return_noise_score=False):
        B, T = idx.shape
        x = self.base.tok_embed(idx) * math.sqrt(self.base.d_model)
        x = self.base.drop(x)

        noise_score_out = None
        for i, block in enumerate(self.base.blocks):
            # Attention residual (standard)
            x = x + block.attn(block.norm1(x), mask=self.base.mask)

            # FFN
            normed = block.norm2(x)
            ffn_out = block.ffn(normed)

            if i == self.disc_layer:
                ffn_out, noise_score_out = self.discriminator(ffn_out)

            x = x + ffn_out

        x = self.base.norm(x)
        logits = self.base.head(x)

        if return_noise_score:
            return logits, noise_score_out
        return logits

    def trainable_parameters(self):
        """Return only discriminator parameters."""
        return self.discriminator.parameters()

    def disc_net_parameters(self):
        """Return discriminator network parameters (not gate_scale)."""
        return list(self.discriminator.net.parameters())

    def gate_parameters(self):
        """Return only gate_scale parameter."""
        return [self.discriminator.gate_scale]


# ============================================================
# Evaluation (Group G specific - includes disc metrics)
# ============================================================

@torch.no_grad()
def evaluate_groupG(model, val_data, max_batches=50, batch_size=16):
    model.eval()
    total_loss = 0
    total_tokens = 0
    all_wrong_confs = []
    all_bins = np.zeros(10)
    all_bin_correct = np.zeros(10)
    all_bin_conf = np.zeros(10)

    # Discriminator-specific metrics
    noise_scores_correct = []
    noise_scores_incorrect = []
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

        # Predictions
        probs = F.softmax(logits_f, dim=-1)
        conf, pred = probs.max(dim=-1)
        correct = (pred == tgt)
        wrong = ~correct

        # Confidence on errors
        if wrong.any():
            all_wrong_confs.extend(conf[wrong].cpu().numpy())

        # ECE bins
        conf_flat = conf.cpu().numpy().ravel()
        correct_flat = correct.cpu().numpy().ravel().astype(float)
        for b in range(10):
            lo, hi = b / 10, (b + 1) / 10
            mask_bin = (conf_flat > lo) & (conf_flat <= hi)
            all_bins[b] += mask_bin.sum()
            all_bin_correct[b] += correct_flat[mask_bin].sum()
            all_bin_conf[b] += conf_flat[mask_bin].sum()

        # Discriminator metrics
        if noise_score is not None:
            ns_mean = noise_score.mean(dim=-1)  # (B, T) mean across d_model
            if correct.any():
                noise_scores_correct.extend(ns_mean[correct].cpu().numpy())
            if wrong.any():
                noise_scores_incorrect.extend(ns_mean[wrong].cpu().numpy())

            # Disc "accuracy": does it assign higher noise score to wrong predictions?
            threshold = 0.5
            disc_pred_wrong = (ns_mean > threshold)
            disc_correct_count += (disc_pred_wrong == wrong).sum().item()
            disc_total_count += wrong.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20))

    # ECE
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

    ns_correct = np.mean(noise_scores_correct) if noise_scores_correct else 0
    ns_incorrect = np.mean(noise_scores_incorrect) if noise_scores_incorrect else 0
    disc_acc = disc_correct_count / max(disc_total_count, 1)

    gate_val = torch.sigmoid(model.discriminator.gate_scale).item()

    return {
        'loss': avg_loss,
        'perplexity': ppl,
        'ece': ece,
        'conf_on_errors': conf_on_errors,
        'high_conf_error_frac': high_conf_frac,
        'disc_accuracy': disc_acc,
        'mean_ns_correct': ns_correct,
        'mean_ns_incorrect': ns_incorrect,
        'gate_scale': gate_val,
    }


# ============================================================
# Training Loop
# ============================================================

def find_best_groupA_checkpoint():
    """Find Group A checkpoint with lowest perplexity."""
    best_ppl = float('inf')
    best_seed = 42
    for seed in [42, 123, 456]:
        metrics_path = os.path.join(BASE_DIR, f'metrics_log_groupA_seed{seed}.csv')
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                lines = f.readlines()
                if len(lines) > 1:
                    last = lines[-1].strip().split(',')
                    ppl = float(last[3])
                    if ppl < best_ppl:
                        best_ppl = ppl
                        best_seed = seed
    return best_seed, best_ppl


def train_groupG(seed, train_data, val_data, total_steps=5000,
                 warmup_disc_steps=1000, micro_batch=16, accum_steps=2, disc_layer=3):
    effective_batch = micro_batch * accum_steps  # =32, matches Groups A-D
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Find and load best Group A checkpoint
    best_seed_A, best_ppl_A = find_best_groupA_checkpoint()
    ckpt_path = os.path.join(BASE_DIR, 'checkpoints', f'groupA_seed{best_seed_A}_final.pt')
    print(f"\n{'='*60}")
    print(f"Group G | Seed {seed} | Base: Group A seed {best_seed_A} (PPL={best_ppl_A:.1f})")
    print(f"Discriminator at layer {disc_layer}")
    print(f"Effective batch_size={effective_batch} (micro={micro_batch} x accum={accum_steps})")
    print(f"{'='*60}")

    # Load base model
    base_model = TransformerLM(group='A').to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    base_model.load_state_dict(ckpt['model'])
    print(f"  Loaded checkpoint from {ckpt_path}")

    # Create Group G model
    model = GroupGModel(base_model, disc_layer=disc_layer, d_model=384).to(DEVICE)

    trainable_count = sum(p.numel() for p in model.trainable_parameters())
    frozen_count = sum(p.numel() for p in model.base.parameters())
    print(f"  Trainable params: {trainable_count:,} ({trainable_count/frozen_count*100:.1f}% of base)")

    # Separate optimizers for disc net and gate_scale
    optimizer_disc = torch.optim.Adam(model.disc_net_parameters(), lr=1e-4)
    optimizer_gate = torch.optim.Adam(model.gate_parameters(), lr=5e-3)
    scaler = GradScaler('cuda')

    # Logging
    metrics_log_path = os.path.join(BASE_DIR, f'metrics_log_groupG_seed{seed}.csv')
    metrics_writer = open(metrics_log_path, 'w', newline='')
    metrics_csv = csv.writer(metrics_writer)
    metrics_csv.writerow(['step', 'seed', 'group', 'perplexity', 'loss', 'ece',
                          'conf_on_errors', 'high_conf_error_frac',
                          'disc_accuracy', 'mean_ns_correct', 'mean_ns_incorrect', 'gate_scale'])

    n_train = len(train_data)
    step = 0
    micro_count = 0
    running_loss = 0
    t_start = time.time()

    print(f"\n  Phase 1: Discriminator warmup (steps 1-{warmup_disc_steps})")

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
            is_accum_boundary = (micro_count % accum_steps == 0)

            if step + 1 == warmup_disc_steps + 1 and is_accum_boundary:
                print(f"\n  Phase 2: Joint training (steps {warmup_disc_steps+1}-{total_steps})")

            # ============================================
            # Phase 1: Discriminator warmup (no gating)
            # ============================================
            if step < warmup_disc_steps or (step == warmup_disc_steps and not is_accum_boundary):
                model.eval()
                model.discriminator.train()

                with autocast('cuda'):
                    logits, noise_score = model(inp, return_noise_score=True)

                logits_f = logits.float().detach()
                _, pred = logits_f.max(dim=-1)
                correct = (pred == tgt)

                if noise_score is not None:
                    ns = noise_score
                    loss_correct = torch.mean(ns[correct] ** 2) if correct.any() else torch.tensor(0.0, device=DEVICE)
                    loss_incorrect = torch.mean((1 - ns[~correct]) ** 2) if (~correct).any() else torch.tensor(0.0, device=DEVICE)
                    loss_sparse = 0.01 * torch.mean(ns)
                    loss_disc = (loss_correct + loss_incorrect + loss_sparse) / accum_steps

                    loss_disc.backward()
                    running_loss += loss_disc.item() * accum_steps

                if is_accum_boundary:
                    optimizer_disc.step()
                    optimizer_disc.zero_grad()
                    step += 1

            # ============================================
            # Phase 2: Joint training with alternating
            # ============================================
            else:
                cycle_pos = (step - warmup_disc_steps) % 50

                if cycle_pos < 40:
                    model.eval()
                    model.discriminator.gate_scale.requires_grad_(True)
                    for p in model.disc_net_parameters():
                        p.requires_grad_(False)

                    with autocast('cuda'):
                        logits = model(inp)
                    logits_f = logits.float()

                    # Gate objective: minimize confidence on WRONG predictions
                    # Do NOT use CE loss here — it always pushes gate → 0 (no suppression)
                    # since the frozen base model is already CE-optimal.
                    probs = F.softmax(logits_f, dim=-1)
                    conf, pred = probs.max(dim=-1)  # (B, T)
                    wrong = (pred != tgt)
                    if wrong.any():
                        # Primary: reduce confidence on errors
                        loss_conf = conf[wrong].mean()
                        # Penalty: don't suppress correct predictions' confidence too much
                        # (mild regularization to prevent gate from killing everything)
                        if (~wrong).any():
                            loss_preserve = -0.1 * conf[~wrong].mean()  # keep correct confidence
                        else:
                            loss_preserve = torch.tensor(0.0, device=DEVICE)
                        loss = loss_conf + loss_preserve
                    else:
                        loss = torch.tensor(0.0, device=DEVICE)

                    loss_scaled = loss / accum_steps

                    scaler.scale(loss_scaled).backward()
                    running_loss += loss.item()

                    if is_accum_boundary:
                        scaler.unscale_(optimizer_gate)
                        scaler.step(optimizer_gate)
                        scaler.update()
                        optimizer_gate.zero_grad()
                        for p in model.disc_net_parameters():
                            p.requires_grad_(True)
                        step += 1

                else:
                    model.eval()
                    model.discriminator.train()

                    with autocast('cuda'):
                        logits, noise_score = model(inp, return_noise_score=True)

                    logits_f = logits.float().detach()
                    _, pred = logits_f.max(dim=-1)
                    correct = (pred == tgt)

                    if noise_score is not None:
                        ns = noise_score
                        loss_correct = torch.mean(ns[correct] ** 2) if correct.any() else torch.tensor(0.0, device=DEVICE)
                        loss_incorrect = torch.mean((1 - ns[~correct]) ** 2) if (~correct).any() else torch.tensor(0.0, device=DEVICE)
                        loss_sparse = 0.01 * torch.mean(ns)
                        loss_disc = (loss_correct + loss_incorrect + loss_sparse) / accum_steps

                        loss_disc.backward()
                        running_loss += loss_disc.item() * accum_steps

                    if is_accum_boundary:
                        optimizer_disc.step()
                        optimizer_disc.zero_grad()
                        step += 1

            # Only log/eval on actual step boundaries
            if not is_accum_boundary:
                continue

            # ============================================
            # Evaluation every 500 steps
            # ============================================
            if step % 500 == 0 and step > 0:
                avg_loss = running_loss / 500
                running_loss = 0

                metrics = evaluate_groupG(model, val_data, max_batches=50, batch_size=micro_batch)
                metrics_csv.writerow([
                    step, seed, 'G',
                    metrics['perplexity'], metrics['loss'], metrics['ece'],
                    metrics['conf_on_errors'], metrics['high_conf_error_frac'],
                    metrics['disc_accuracy'], metrics['mean_ns_correct'],
                    metrics['mean_ns_incorrect'], metrics['gate_scale'],
                ])
                metrics_writer.flush()

                elapsed = time.time() - t_start
                steps_per_sec = step / elapsed
                eta = (total_steps - step) / steps_per_sec

                phase = "Warmup" if step <= warmup_disc_steps else "Joint"
                gate_val = torch.sigmoid(model.discriminator.gate_scale).item()

                print(f"  Step {step:5d}/{total_steps} [{phase}] | "
                      f"PPL {metrics['perplexity']:.1f} | "
                      f"ConfErr {metrics['conf_on_errors']:.4f} | "
                      f"DiscAcc {metrics['disc_accuracy']:.3f} | "
                      f"NS_corr {metrics['mean_ns_correct']:.3f} / NS_wrong {metrics['mean_ns_incorrect']:.3f} | "
                      f"Gate {gate_val:.4f} | "
                      f"ETA {eta/60:.0f}m")

    metrics_writer.close()

    # Final evaluation
    final_metrics = evaluate_groupG(model, val_data, max_batches=100, batch_size=micro_batch)
    print(f"\n  FINAL: PPL={final_metrics['perplexity']:.1f} | "
          f"ConfErr={final_metrics['conf_on_errors']:.4f} | "
          f"DiscAcc={final_metrics['disc_accuracy']:.3f} | "
          f"Gate={final_metrics['gate_scale']:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(BASE_DIR, 'checkpoints', f'groupG_seed{seed}_final.pt')
    torch.save({
        'discriminator': model.discriminator.state_dict(),
        'disc_layer': disc_layer,
        'base_seed': find_best_groupA_checkpoint()[0],
        'seed': seed,
        'step': step,
        'final_metrics': final_metrics,
    }, ckpt_path)

    del model
    torch.cuda.empty_cache()

    return final_metrics


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 2 GROUP G: Adversarial Noise Discriminator")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print("=" * 60)

    # Verify Group A checkpoints exist
    best_seed, best_ppl = find_best_groupA_checkpoint()
    ckpt_path = os.path.join(BASE_DIR, 'checkpoints', f'groupA_seed{best_seed}_final.pt')
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Group A checkpoint not found at {ckpt_path}")
        print("Please complete Groups A-D first.")
        return

    print(f"Using Group A seed {best_seed} (PPL={best_ppl:.1f}) as base model")

    train_data, val_data = load_data(ctx_len=256)

    seeds = [42, 123, 456]
    all_results = {}

    for seed in seeds:
        key = f"G_{seed}"
        result = train_groupG(
            seed=seed,
            train_data=train_data,
            val_data=val_data,
            total_steps=5000,
            warmup_disc_steps=1000,
            micro_batch=16,
            accum_steps=2,
            disc_layer=3,
        )
        all_results[key] = result

    # Summary
    print("\n" + "=" * 60)
    print("GROUP G COMPLETE")
    print("=" * 60)

    ppls = [all_results[f"G_{s}"]['perplexity'] for s in seeds]
    confs = [all_results[f"G_{s}"]['conf_on_errors'] for s in seeds]
    disc_accs = [all_results[f"G_{s}"]['disc_accuracy'] for s in seeds]
    gates = [all_results[f"G_{s}"]['gate_scale'] for s in seeds]

    print(f"  PPL:        {np.mean(ppls):.1f} ± {np.std(ppls):.1f}")
    print(f"  ConfOnErr:  {np.mean(confs):.4f} ± {np.std(confs):.4f}")
    print(f"  DiscAcc:    {np.mean(disc_accs):.3f} ± {np.std(disc_accs):.3f}")
    print(f"  GateScale:  {np.mean(gates):.4f} ± {np.std(gates):.4f}")

    # Save summary
    summary_path = os.path.join(BASE_DIR, 'groupG_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'mean_ppl': float(np.mean(ppls)),
            'std_ppl': float(np.std(ppls)),
            'mean_conf_on_errors': float(np.mean(confs)),
            'std_conf_on_errors': float(np.std(confs)),
            'mean_disc_accuracy': float(np.mean(disc_accs)),
            'mean_gate_scale': float(np.mean(gates)),
            'per_seed': {str(s): all_results[f"G_{s}"] for s in seeds},
        }, f, indent=2, default=str)

    print(f"\n  Summary saved to {summary_path}")


if __name__ == '__main__':
    main()
