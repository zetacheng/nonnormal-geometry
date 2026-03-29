"""
Phase 2 EFG: Targeted Noise Suppression Experiments
Groups E (Perturbation Consistency), F (Schur Projection), G (Adversarial Discriminator)
All fine-tuned from frozen Group A checkpoints.
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

# Import base components from phase2_lm
from phase2_lm import (
    TransformerLM, evaluate, load_data, measure_geometry,
    DEVICE, BASE_DIR
)

EFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', 'phase2_lm')
os.makedirs(os.path.join(EFG_DIR, 'checkpoints'), exist_ok=True)


# ============================================================
# Utility: Load Frozen Group A Base Model
# ============================================================

def load_frozen_base(seed):
    """Load a trained Group A checkpoint with all params frozen."""
    model = TransformerLM(group='A').to(DEVICE)
    ckpt_path = os.path.join(EFG_DIR, 'checkpoints', f'groupA_seed{seed}_final.pt')
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    print(f"  Loaded frozen Group A (seed {seed}) from {ckpt_path}")
    return model


# ============================================================
# GROUP E: Perturbation Consistency Gating
# ============================================================

class ConsistencyGateMLP(nn.Module):
    """Per-layer gate MLP: takes consistency scores, outputs gate values."""
    def __init__(self, d_model, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, consistency):
        return torch.sigmoid(self.fc2(F.relu(self.fc1(consistency))))


class ConsistencyGatedLM(nn.Module):
    """Group E: Perturbation consistency gating wrapper."""
    def __init__(self, base_model, perturb_scale=0.01):
        super().__init__()
        self.base = base_model  # frozen
        self.perturb_scale = perturb_scale
        d = base_model.d_model
        n_layers = len(base_model.blocks)

        self.gate_mlps = nn.ModuleList([
            ConsistencyGateMLP(d, hidden=16) for _ in range(n_layers)
        ])

        # For logging
        self.last_consistency = [None] * n_layers

    def forward(self, idx):
        B, T = idx.shape
        x = self.base.tok_embed(idx) * math.sqrt(self.base.d_model)
        x = self.base.drop(x)

        for i, block in enumerate(self.base.blocks):
            # Attention (standard, frozen)
            x = x + block.attn(block.norm1(x), mask=self.base.mask)

            # FFN with consistency gating
            normed = block.norm2(x)
            h1 = block.ffn(normed)

            with torch.no_grad():
                noise = torch.randn_like(normed) * self.perturb_scale * normed.norm(dim=-1, keepdim=True)
                h2 = block.ffn(normed + noise)

            # Consistency: how stable is each dimension?
            consistency = (1.0 - (h1 - h2).abs() / (h1.abs() + 1e-8)).detach()
            consistency = consistency.clamp(0, 1)

            # Gate
            gate = self.gate_mlps[i](consistency)
            x = x + h1 * gate

            # Store for logging
            self.last_consistency[i] = consistency.mean().item()

            # Store alpha/beta for compatibility with measure_geometry
            block.last_alpha = gate.mean().item()
            block.last_beta = 0.0

        x = self.base.norm(x)
        return self.base.head(x)


# ============================================================
# GROUP F: Schur Direction Projection
# ============================================================

def compute_schur_init(base_model, k=16):
    """Compute Schur decomposition of each layer's FFN effective operator
    and extract top-k non-normal direction columns."""
    U_noise_list = []
    d = base_model.d_model

    for i, block in enumerate(base_model.blocks):
        W1 = block.ffn.w1.weight.detach().float().cpu()  # (d_ff, d_model)
        W2 = block.ffn.w2.weight.detach().float().cpu()  # (d_model, d_ff)
        M = W2 @ W1  # (d_model, d_model)

        # Schur decomposition: M = U T U*
        T_schur, U = torch.linalg.schur(M)

        # Non-normality per column: norm of strict upper-triangular entries
        col_norms = torch.zeros(T_schur.shape[1])
        for j in range(T_schur.shape[1]):
            if j > 0:
                col_norms[j] = T_schur[:j, j].norm().item()

        # Top-k columns with highest non-normal contribution
        topk_idx = col_norms.topk(min(k, len(col_norms))).indices
        U_init = U[:, topk_idx].to(DEVICE)  # (d_model, k)

        # Orthonormalize (Schur vectors should already be orthonormal, but be safe)
        U_init, _ = torch.linalg.qr(U_init)

        U_noise_list.append(U_init)
        print(f"  Layer {i}: top non-normal col norms = {col_norms[topk_idx[:5]].tolist()}")

    return U_noise_list


class SchurProjectionLM(nn.Module):
    """Group F: Project away from non-normal Schur directions."""
    def __init__(self, base_model, k=16):
        super().__init__()
        self.base = base_model  # frozen
        d = base_model.d_model
        n_layers = len(base_model.blocks)

        # Initialize from Schur decomposition
        print("  Computing Schur initialization...")
        U_inits = compute_schur_init(base_model, k=k)

        self.U_noise = nn.ParameterList([
            nn.Parameter(U_inits[i].clone()) for i in range(n_layers)
        ])
        self.suppress_scale = nn.ParameterList([
            nn.Parameter(torch.tensor(0.0, device=DEVICE)) for _ in range(n_layers)
        ])

        # Store initial directions for comparison
        self.register_buffer('U_noise_init_0', U_inits[0].clone())

    def forward(self, idx):
        B, T = idx.shape
        x = self.base.tok_embed(idx) * math.sqrt(self.base.d_model)
        x = self.base.drop(x)

        for i, block in enumerate(self.base.blocks):
            # Attention (standard, frozen)
            x = x + block.attn(block.norm1(x), mask=self.base.mask)

            # FFN with Schur projection
            normed = block.norm2(x)
            h = block.ffn(normed)

            # Project away from non-normal directions
            U = self.U_noise[i]  # (d_model, k)
            proj = h @ U @ U.T  # (B, T, d_model)
            scale = torch.sigmoid(self.suppress_scale[i])  # keep scale bounded [0,1]
            h_clean = h - scale * proj

            x = x + h_clean

            block.last_alpha = (1.0 - scale.item())
            block.last_beta = 0.0

        x = self.base.norm(x)
        return self.base.head(x)


# ============================================================
# GROUP G: Adversarial Noise Discriminator
# ============================================================

class NoiseDiscriminator(nn.Module):
    """Small MLP that predicts per-dimension noise scores."""
    def __init__(self, d_model, d_hidden=64):
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

    def forward(self, h, suppress=True):
        noise_score = self.net(h.detach())
        if suppress:
            gate = 1.0 - torch.sigmoid(self.gate_scale) * noise_score
            return h * gate, noise_score
        return noise_score


class AdversarialDiscriminatorLM(nn.Module):
    """Group G: Adversarial noise discriminator at middle layer."""
    def __init__(self, base_model, disc_layer=3):
        super().__init__()
        self.base = base_model  # frozen
        self.disc_layer = disc_layer
        d = base_model.d_model
        self.discriminator = NoiseDiscriminator(d, d_hidden=64)
        self.last_noise_score_mean = 0.0

    def forward(self, idx, suppress=True):
        B, T = idx.shape
        x = self.base.tok_embed(idx) * math.sqrt(self.base.d_model)
        x = self.base.drop(x)

        noise_score = None
        for i, block in enumerate(self.base.blocks):
            # Attention
            x = x + block.attn(block.norm1(x), mask=self.base.mask)
            # FFN
            normed = block.norm2(x)
            h = block.ffn(normed)

            if i == self.disc_layer and suppress:
                h, noise_score = self.discriminator(h, suppress=True)
                self.last_noise_score_mean = noise_score.mean().item()

            x = x + h

            block.last_alpha = 1.0
            block.last_beta = 0.0

        x = self.base.norm(x)
        logits = self.base.head(x)
        return logits, noise_score

    def forward_for_eval(self, idx):
        """Evaluate-compatible forward (returns logits only)."""
        logits, _ = self.forward(idx, suppress=True)
        return logits


# ============================================================
# Modified evaluate for Group G (wraps the standard evaluate)
# ============================================================

@torch.no_grad()
def evaluate_g(model, val_data, max_batches=50, batch_size=16):
    """Evaluate Group G model (which returns logits, noise_score)."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    all_wrong_confs = []
    all_bins = np.zeros(10)
    all_bin_correct = np.zeros(10)
    all_bin_conf = np.zeros(10)

    n_batches = min(max_batches, len(val_data) // batch_size)

    for i in range(n_batches):
        x = val_data[i * batch_size:(i + 1) * batch_size].to(DEVICE)
        inp = x[:, :-1]
        tgt = x[:, 1:]

        with autocast('cuda'):
            logits = model.forward_for_eval(inp)
        logits_f = logits.float()
        loss = F.cross_entropy(logits_f.view(-1, logits_f.size(-1)), tgt.reshape(-1))
        total_loss += loss.item() * tgt.numel()
        total_tokens += tgt.numel()

        probs = F.softmax(logits_f, dim=-1)
        conf, pred = probs.max(dim=-1)
        wrong = (pred != tgt)
        if wrong.any():
            all_wrong_confs.extend(conf[wrong].cpu().numpy())

        conf_flat = conf.cpu().numpy().ravel()
        correct_flat = (pred == tgt).cpu().numpy().ravel().astype(float)
        for b in range(10):
            lo, hi = b / 10, (b + 1) / 10
            mask = (conf_flat > lo) & (conf_flat <= hi)
            all_bins[b] += mask.sum()
            all_bin_correct[b] += correct_flat[mask].sum()
            all_bin_conf[b] += conf_flat[mask].sum()

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

    return {
        'loss': avg_loss,
        'perplexity': ppl,
        'ece': ece,
        'conf_on_errors': conf_on_errors,
        'high_conf_error_frac': high_conf_frac,
    }


# ============================================================
# Discriminator Training Loss for Group G
# ============================================================

def discriminator_loss(model, x, lambda_sparse=0.01):
    """Compute discriminator loss: identify noise dimensions on incorrect tokens."""
    inp = x[:, :-1]
    tgt = x[:, 1:]

    with autocast('cuda'):
        logits, noise_score = model(inp, suppress=False)

    logits_f = logits.float()
    pred = logits_f.argmax(dim=-1)
    correct = (pred == tgt).float()  # (B, T)
    incorrect = 1.0 - correct

    if noise_score is None:
        return torch.tensor(0.0, device=DEVICE)

    # For correct tokens: noise_score should be low
    if correct.sum() > 0:
        l_correct = (noise_score * correct.unsqueeze(-1)).pow(2).sum() / max(correct.sum(), 1)
    else:
        l_correct = torch.tensor(0.0, device=DEVICE)

    # For incorrect tokens: noise_score should be high (at least some dimensions)
    if incorrect.sum() > 0:
        l_incorrect = -(noise_score * incorrect.unsqueeze(-1)).pow(2).sum() / max(incorrect.sum(), 1)
    else:
        l_incorrect = torch.tensor(0.0, device=DEVICE)

    # Sparsity: don't flag too many dimensions
    l_sparse = noise_score.abs().mean() * lambda_sparse

    return l_correct + l_incorrect + l_sparse


# ============================================================
# Unified Training Loop for Groups E, F, G
# ============================================================

def train_efg(group, seed, train_data, val_data, total_steps=5000,
              batch_size=32, lr=3e-4, warmup_steps=200):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    group_names = {'E': 'ConsistencyGate', 'F': 'SchurProjection', 'G': 'AdversarialDisc'}
    print(f"\n{'='*60}")
    print(f"Group {group}: {group_names[group]} | Seed {seed}")
    print(f"{'='*60}")

    # Load frozen base
    base_model = load_frozen_base(seed)

    # Create wrapper model
    if group == 'E':
        model = ConsistencyGatedLM(base_model).to(DEVICE)
    elif group == 'F':
        model = SchurProjectionLM(base_model, k=16).to(DEVICE)
    elif group == 'G':
        model = AdversarialDiscriminatorLM(base_model, disc_layer=3).to(DEVICE)
    else:
        raise ValueError(f"Unknown group: {group}")

    # Count trainable params
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    print(f"  Trainable params: {n_trainable:,} (base frozen)")

    # Optimizer (only trainable params)
    if group == 'G':
        # Separate optimizers for discriminator and gate_scale
        disc_params = list(model.discriminator.net.parameters())
        gate_params = [model.discriminator.gate_scale]
        main_optimizer = torch.optim.AdamW(gate_params, lr=lr, weight_decay=0.0)
        disc_optimizer = torch.optim.AdamW(disc_params, lr=lr * 0.3, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)

    scaler = GradScaler('cuda')

    # LR schedule
    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    if group == 'G':
        main_scheduler = torch.optim.lr_scheduler.LambdaLR(main_optimizer, lr_schedule)
        disc_scheduler = torch.optim.lr_scheduler.LambdaLR(disc_optimizer, lr_schedule)
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Logging
    metrics_log_path = os.path.join(EFG_DIR, f'metrics_log_group{group}_seed{seed}.csv')
    extra_log_path = os.path.join(EFG_DIR, f'extra_log_group{group}_seed{seed}.csv')

    metrics_writer = open(metrics_log_path, 'w', newline='')
    metrics_csv = csv.writer(metrics_writer)
    metrics_csv.writerow(['step', 'seed', 'group', 'perplexity', 'loss', 'ece',
                          'conf_on_errors', 'high_conf_error_frac'])

    extra_writer = open(extra_log_path, 'w', newline='')
    extra_csv = csv.writer(extra_writer)
    if group == 'E':
        extra_csv.writerow(['step'] + [f'consistency_L{i}' for i in range(6)])
    elif group == 'F':
        extra_csv.writerow(['step'] + [f'suppress_scale_L{i}' for i in range(6)] +
                           [f'angle_drift_L{i}' for i in range(6)])
    elif group == 'G':
        extra_csv.writerow(['step', 'gate_scale', 'mean_noise_score', 'disc_loss'])

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

            idx = perm[i:i + batch_size]
            x = train_data[idx].to(DEVICE)
            inp = x[:, :-1]
            tgt = x[:, 1:]

            # ---- Group G: Alternating schedule ----
            if group == 'G':
                is_disc_step = (step >= 500) and (step % 50 >= 40)

                if is_disc_step and step >= 500:
                    # Discriminator update step
                    disc_optimizer.zero_grad()
                    d_loss = discriminator_loss(model, x)
                    d_loss.backward()
                    disc_optimizer.step()
                    disc_scheduler.step()
                    running_loss += d_loss.item()
                else:
                    # Main model step (train gate_scale with CE loss)
                    main_optimizer.zero_grad()
                    with autocast('cuda'):
                        logits = model.forward_for_eval(inp)
                        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.reshape(-1))
                    scaler.scale(loss).backward()
                    scaler.step(main_optimizer)
                    scaler.update()
                    main_scheduler.step()
                    running_loss += loss.item()

            # ---- Groups E, F: Standard training ----
            else:
                model.train()
                optimizer.zero_grad()

                with autocast('cuda'):
                    logits = model(inp)
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.reshape(-1))

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running_loss += loss.item()

            step += 1

            # Extra logging every 250 steps
            if step % 250 == 0:
                if group == 'E':
                    row = [step] + [model.last_consistency[j] or 0 for j in range(6)]
                    extra_csv.writerow(row)
                elif group == 'F':
                    scales = [torch.sigmoid(model.suppress_scale[j]).item() for j in range(6)]
                    # Angle drift from initial Schur directions (only layer 0 tracked)
                    angles = [0.0] * 6
                    try:
                        U_curr = model.U_noise[0].detach()
                        U_init = model.U_noise_init_0
                        cos_sim = torch.trace(U_curr.T @ U_init) / (U_curr.norm() * U_init.norm())
                        angles[0] = math.degrees(math.acos(min(max(cos_sim.item(), -1), 1)))
                    except:
                        pass
                    row = [step] + scales + angles
                    extra_csv.writerow(row)
                elif group == 'G':
                    gs = torch.sigmoid(model.discriminator.gate_scale).item()
                    ns = model.last_noise_score_mean
                    row = [step, gs, ns, running_loss / max(step, 1)]
                    extra_csv.writerow(row)
                extra_writer.flush()

            # Evaluation every 500 steps
            if step % 500 == 0:
                avg_train_loss = running_loss / 500
                running_loss = 0

                if group == 'G':
                    metrics = evaluate_g(model, val_data, max_batches=50, batch_size=batch_size)
                else:
                    metrics = evaluate(model, val_data, max_batches=50, batch_size=batch_size)

                metrics_csv.writerow([step, seed, group, metrics['perplexity'],
                                      metrics['loss'], metrics['ece'],
                                      metrics['conf_on_errors'], metrics['high_conf_error_frac']])
                metrics_writer.flush()

                elapsed = time.time() - t_start
                steps_per_sec = step / elapsed
                eta = (total_steps - step) / steps_per_sec

                print(f"  Step {step:5d}/{total_steps} | "
                      f"TrainLoss {avg_train_loss:.4f} | "
                      f"ValPPL {metrics['perplexity']:.1f} | "
                      f"ECE {metrics['ece']:.4f} | "
                      f"ConfErr {metrics['conf_on_errors']:.3f} | "
                      f"ETA {eta/60:.0f}m")

            # Checkpoint every 2500 steps
            if step % 2500 == 0:
                ckpt_path = os.path.join(EFG_DIR, 'checkpoints',
                                         f'group{group}_seed{seed}_step{step}.pt')
                save_dict = {
                    'step': step,
                    'group': group,
                    'seed': seed,
                    'base_checkpoint': f'groupA_seed{seed}_final.pt',
                }
                if group == 'E':
                    save_dict['gate_mlps'] = model.gate_mlps.state_dict()
                elif group == 'F':
                    save_dict['U_noise'] = model.U_noise.state_dict()
                    save_dict['suppress_scale'] = model.suppress_scale.state_dict()
                elif group == 'G':
                    save_dict['discriminator'] = model.discriminator.state_dict()
                torch.save(save_dict, ckpt_path)

    metrics_writer.close()
    extra_writer.close()

    # Final evaluation
    if group == 'G':
        final_metrics = evaluate_g(model, val_data, max_batches=100, batch_size=batch_size)
    else:
        final_metrics = evaluate(model, val_data, max_batches=100, batch_size=batch_size)

    print(f"\n  FINAL: PPL={final_metrics['perplexity']:.1f} | "
          f"ECE={final_metrics['ece']:.4f} | "
          f"ConfErr={final_metrics['conf_on_errors']:.3f}")

    # Save final checkpoint
    ckpt_path = os.path.join(EFG_DIR, 'checkpoints', f'group{group}_seed{seed}_final.pt')
    save_dict = {
        'step': step,
        'group': group,
        'seed': seed,
        'final_metrics': final_metrics,
        'base_checkpoint': f'groupA_seed{seed}_final.pt',
    }
    if group == 'E':
        save_dict['gate_mlps'] = model.gate_mlps.state_dict()
    elif group == 'F':
        save_dict['U_noise'] = model.U_noise.state_dict()
        save_dict['suppress_scale'] = model.suppress_scale.state_dict()
    elif group == 'G':
        save_dict['discriminator'] = model.discriminator.state_dict()
    torch.save(save_dict, ckpt_path)

    del model, base_model
    torch.cuda.empty_cache()

    return final_metrics


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 2 EFG: Targeted Noise Suppression Experiments")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print("=" * 60)

    train_data, val_data = load_data(ctx_len=256)

    groups = ['E', 'F', 'G']
    seeds = [42, 123, 456]
    total_steps = 5000

    all_results = {}

    for group in groups:
        for seed in seeds:
            key = f"{group}_{seed}"

            # Check if Group A checkpoint exists
            ckpt_path = os.path.join(EFG_DIR, 'checkpoints', f'groupA_seed{seed}_final.pt')
            if not os.path.exists(ckpt_path):
                print(f"\n  [SKIP] {key}: Group A checkpoint not found at {ckpt_path}")
                continue

            result = train_efg(
                group=group, seed=seed,
                train_data=train_data, val_data=val_data,
                total_steps=total_steps,
            )
            all_results[key] = result

            # Save intermediate summary
            summary_path = os.path.join(EFG_DIR, 'results_summary_efg.json')
            with open(summary_path, 'w') as f:
                json.dump({k: v for k, v in all_results.items()}, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("PHASE 2 EFG COMPLETE")
    print("=" * 60)

    group_names = {'E': 'ConsistencyGate', 'F': 'SchurProjection', 'G': 'AdversarialDisc'}
    for group in groups:
        ppls = [all_results[f"{group}_{s}"]['perplexity'] for s in seeds
                if f"{group}_{s}" in all_results]
        eces = [all_results[f"{group}_{s}"]['ece'] for s in seeds
                if f"{group}_{s}" in all_results]
        confs = [all_results[f"{group}_{s}"]['conf_on_errors'] for s in seeds
                 if f"{group}_{s}" in all_results]
        if ppls:
            print(f"  Group {group} ({group_names[group]}): "
                  f"PPL={np.mean(ppls):.1f}+/-{np.std(ppls):.1f} | "
                  f"ECE={np.mean(eces):.4f} | "
                  f"ConfErr={np.mean(confs):.3f}")


if __name__ == '__main__':
    main()
