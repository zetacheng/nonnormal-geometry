"""
Phase 4: Hallucination via Noise Injection
Inject structured noise matrices into a trained Group A model's FFN output
and measure how different noise geometries affect calibration and accuracy.
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import json
import csv
import math
import numpy as np
import pandas as pd
from scipy.linalg import expm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# --- paths ---
PAPER_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PAPER_DIR)
from phase2_lm import TransformerLM, load_data, RMSNorm, DEVICE

DATA_DIR = os.path.join(PAPER_DIR, 'experiments', 'phase2_lm')
CKPT_DIR = os.path.join(DATA_DIR, 'checkpoints')
OUT_DIR = os.path.join(PAPER_DIR, 'experiments', 'phase4_hallucination')
os.makedirs(OUT_DIR, exist_ok=True)

torch.cuda.set_per_process_memory_fraction(0.90, 0)

D_MODEL = 384
INJECT_LAYER = 3
STRENGTHS = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


# ================================================================
# Noise Matrix Construction
# ================================================================

def make_growth_noise(d=D_MODEL, seed=0):
    """Type A: Upper-triangular random (transient growth inducing)."""
    rng = np.random.RandomState(seed)
    M = np.triu(rng.randn(d, d))
    # Normalize to operator norm 1
    op_norm = np.linalg.norm(M, ord=2)
    if op_norm > 1e-12:
        M = M / op_norm
    return M


def make_no_growth_noise(d=D_MODEL, seed=0):
    """Type B: Strongly stable diagonal + weak off-diagonal (no transient growth)."""
    rng = np.random.RandomState(seed)
    # Diagonal: negative real parts -> stable
    diag = -np.abs(rng.randn(d)) - 0.5  # all negative, magnitude >= 0.5
    M = np.diag(diag)
    # Weak off-diagonal
    off_diag = 0.01 * rng.randn(d, d)
    np.fill_diagonal(off_diag, 0)
    M = M + off_diag
    # Normalize to operator norm 1
    op_norm = np.linalg.norm(M, ord=2)
    if op_norm > 1e-12:
        M = M / op_norm
    return M


def make_normal_noise(d=D_MODEL, seed=0):
    """Type C: Symmetric random (normal matrix)."""
    rng = np.random.RandomState(seed)
    R = rng.randn(d, d)
    M = (R + R.T) / 2
    # Normalize to operator norm 1
    op_norm = np.linalg.norm(M, ord=2)
    if op_norm > 1e-12:
        M = M / op_norm
    return M


def analyze_noise_matrix(M, name):
    """Compute Henrici, curvature, transient growth ratio for a noise matrix."""
    d = M.shape[0]
    fro_sq = np.sum(M ** 2)
    eigs = np.linalg.eigvals(M)
    spec_sq = np.sum(np.abs(eigs) ** 2)
    henrici = np.sqrt(max(fro_sq - spec_sq, 0)) / np.sqrt(max(fro_sq, 1e-12))

    H = (M + M.T) / 2
    K = (M - M.T) / 2
    comm = H @ K - K @ H
    curvature = np.linalg.norm(comm, 'fro')

    # Transient growth ratio: max_t ||exp(tM)|| / ||exp(tM) at t=inf||
    # For practical computation, sample a few t values
    max_growth = 1.0
    for t in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]:
        eM = expm(t * M)
        growth = np.linalg.norm(eM, ord=2)
        if growth > max_growth:
            max_growth = growth

    print(f"  {name}:")
    print(f"    Henrici = {henrici:.4f}")
    print(f"    Curvature = {curvature:.4f}")
    print(f"    Transient growth ratio = {max_growth:.4f}")
    print(f"    Spectral radius = {np.max(np.abs(eigs)):.4f}")

    return {'henrici': henrici, 'curvature': curvature, 'transient_growth': max_growth}


# ================================================================
# Hooked Evaluation
# ================================================================

@torch.no_grad()
def evaluate_with_noise(model, val_data, noise_matrix_torch, strength,
                        max_batches=50, batch_size=16):
    """Evaluate model with noise injected at layer INJECT_LAYER FFN output."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    total_correct = 0
    all_wrong_confs = []
    all_bins = np.zeros(10)
    all_bin_correct = np.zeros(10)
    all_bin_conf = np.zeros(10)

    # Hook storage
    hook_handle = None

    def inject_hook(module, input, output):
        """Hook applied to the FFN module of the target layer."""
        # output shape: (B, T, d_model)
        if strength > 0:
            # h = h + strength * (M @ h^T)^T
            # M is (d, d), h is (B, T, d) -> reshape for matmul
            B, T, D = output.shape
            h_flat = output.reshape(-1, D)  # (B*T, D)
            noise_out = h_flat @ noise_matrix_torch.T  # (B*T, D)
            return output + strength * noise_out.reshape(B, T, D)
        return output

    # Register hook on the target layer's FFN
    target_ffn = model.blocks[INJECT_LAYER].ffn
    hook_handle = target_ffn.register_forward_hook(inject_hook)

    n_batches = min(max_batches, len(val_data) // batch_size)

    try:
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

            # Accuracy
            probs = F.softmax(logits_f, dim=-1)
            conf, pred = probs.max(dim=-1)
            correct = (pred == tgt)
            total_correct += correct.sum().item()

            # Confidence on errors
            wrong = ~correct
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
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 20))
    accuracy = total_correct / max(total_tokens, 1)

    # ECE
    ece = 0
    total_bin_count = all_bins.sum()
    for b in range(10):
        if all_bins[b] > 0:
            acc_b = all_bin_correct[b] / all_bins[b]
            conf_b = all_bin_conf[b] / all_bins[b]
            ece += all_bins[b] * abs(acc_b - conf_b)
    ece /= max(total_bin_count, 1)

    wrong_confs = np.array(all_wrong_confs) if all_wrong_confs else np.array([0.0])
    conf_on_errors = wrong_confs.mean()
    high_conf_frac = (wrong_confs > 0.8).mean()

    return {
        'perplexity': ppl,
        'accuracy': accuracy,
        'conf_on_errors': conf_on_errors,
        'high_conf_error_frac': high_conf_frac,
        'ece': ece,
    }


# ================================================================
# Main
# ================================================================

def main():
    print("=" * 70)
    print("PHASE 4: Hallucination via Noise Injection")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}")
    print("=" * 70)

    # --- Load model ---
    ckpt_path = os.path.join(CKPT_DIR, 'groupA_seed42_final.pt')
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model = TransformerLM(group='A').to(DEVICE)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print("  Model loaded.")

    # --- Load validation data ---
    print("\nLoading validation data...")
    _, val_data = load_data(ctx_len=256)
    print(f"  Val sequences: {val_data.shape[0]}")

    # --- Construct noise matrices ---
    print("\nConstructing noise matrices (d={})...".format(D_MODEL))
    noise_matrices = {
        'growth': make_growth_noise(D_MODEL, seed=42),
        'no_growth': make_no_growth_noise(D_MODEL, seed=42),
        'normal': make_normal_noise(D_MODEL, seed=42),
    }

    noise_props = {}
    for name, M in noise_matrices.items():
        noise_props[name] = analyze_noise_matrix(M, name)

    # Convert to torch tensors on GPU
    noise_tensors = {}
    for name, M in noise_matrices.items():
        noise_tensors[name] = torch.tensor(M, dtype=torch.float32).to(DEVICE)

    # --- Sweep ---
    print(f"\nRunning noise injection sweep...")
    print(f"  Layer: {INJECT_LAYER}")
    print(f"  Strengths: {STRENGTHS}")
    print(f"  Noise types: {list(noise_matrices.keys())}")

    results = []
    for noise_name in ['growth', 'no_growth', 'normal']:
        M_t = noise_tensors[noise_name]
        for strength in STRENGTHS:
            print(f"\n  {noise_name}, strength={strength}")
            metrics = evaluate_with_noise(
                model, val_data, M_t, strength,
                max_batches=50, batch_size=16,
            )
            print(f"    PPL={metrics['perplexity']:.2f}, "
                  f"Acc={metrics['accuracy']:.4f}, "
                  f"ConfErr={metrics['conf_on_errors']:.4f}, "
                  f"HiConf={metrics['high_conf_error_frac']:.4f}")

            results.append({
                'noise_type': noise_name,
                'strength': strength,
                'perplexity': metrics['perplexity'],
                'accuracy': metrics['accuracy'],
                'conf_on_errors': metrics['conf_on_errors'],
                'high_conf_frac': metrics['high_conf_error_frac'],
                'ece': metrics['ece'],
            })

            # Clear CUDA cache between runs
            torch.cuda.empty_cache()

    # --- Save CSV ---
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUT_DIR, 'noise_injection_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # Also save noise properties
    props_path = os.path.join(OUT_DIR, 'noise_matrix_properties.json')
    with open(props_path, 'w') as f:
        json.dump(noise_props, f, indent=2, default=str)
    print(f"  Saved: {props_path}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    colors = {'growth': '#d62728', 'no_growth': '#2ca02c', 'normal': '#1f77b4'}
    labels = {'growth': 'Growth (upper-tri)', 'no_growth': 'No-growth (stable diag)',
              'normal': 'Normal (symmetric)'}

    for noise_name in ['growth', 'no_growth', 'normal']:
        subset = df[df['noise_type'] == noise_name]

        # Subplot 1: Accuracy vs strength
        axes[0].plot(subset['strength'], subset['accuracy'], 'o-',
                     color=colors[noise_name], label=labels[noise_name], markersize=4)

        # Subplot 2: Confidence on errors vs strength
        axes[1].plot(subset['strength'], subset['conf_on_errors'], 'o-',
                     color=colors[noise_name], label=labels[noise_name], markersize=4)

    axes[0].set_xlabel('Noise Strength')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()

    axes[1].set_xlabel('Noise Strength')
    axes[1].set_ylabel('Confidence on Errors')
    axes[1].legend()

    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, 'noise_injection_curves.pdf')
    fig.savefig(fig_path)
    plt.close(fig)
    print(f"  Saved: {fig_path}")

    # Cleanup
    del model
    torch.cuda.empty_cache()

    print("\n" + "=" * 70)
    print("PHASE 4 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
