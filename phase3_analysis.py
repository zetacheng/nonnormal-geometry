"""
Phase 3: Post-Training Analysis
Reads logged metrics/geometry, loads checkpoints for detailed analysis.
Produces tables, curvature plots, beta analysis, pseudospectra.
"""
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import sys
import json
import csv
import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import expm

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from phase2_lm import TransformerLM, measure_geometry, RMSNorm

DATA_DIR = os.path.join(PAPER_DIR, 'experiments', 'phase2_lm')
CKPT_DIR = os.path.join(DATA_DIR, 'checkpoints')
OUT_DIR = os.path.join(PAPER_DIR, 'experiments', 'phase3_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
GROUPS = ['A', 'B', 'C', 'D', 'H']
SEEDS = [42, 123, 456]
GROUP_NAMES = {
    'A': 'Standard', 'B': 'PRC ($\\beta\\geq 0$)',
    'C': 'PRC (full $\\beta$)', 'D': 'ForcedNormal',
    'G': 'Post-hoc Disc.', 'H': 'Training Disc.',
}


# ================================================================
# Helpers
# ================================================================

def load_metrics_csv(group, seed):
    """Load a metrics CSV into a DataFrame."""
    path = os.path.join(DATA_DIR, f'metrics_log_group{group}_seed{seed}.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_geometry_csv(group, seed):
    """Load a geometry CSV into a DataFrame."""
    path = os.path.join(DATA_DIR, f'geometry_log_group{group}_seed{seed}.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def load_checkpoint(group, seed, tag='final'):
    """Load a checkpoint, return state_dict."""
    path = os.path.join(CKPT_DIR, f'group{group}_seed{seed}_{tag}.pt')
    if not os.path.exists(path):
        print(f"  [WARN] Checkpoint not found: {path}")
        return None
    return torch.load(path, map_location='cpu', weights_only=False)


def cohens_d(a, b):
    """Compute Cohen's d (pooled SD)."""
    na, nb = len(a), len(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled < 1e-12:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


# ================================================================
# 3A: Main Results Table
# ================================================================

def section_3a():
    print("\n" + "=" * 70)
    print("3A: Main Results Table")
    print("=" * 70)

    # Collect final-row metrics for each group/seed
    records = {}  # group -> {metric: [values]}
    for g in GROUPS:
        records[g] = {'perplexity': [], 'ece': [], 'conf_on_errors': [], 'high_conf_error_frac': []}
        for s in SEEDS:
            df = load_metrics_csv(g, s)
            if df is None:
                # Try groupH_summary.json fallback
                summary_path = os.path.join(DATA_DIR, f'group{g}_summary.json')
                if os.path.exists(summary_path):
                    with open(summary_path) as f:
                        summary = json.load(f)
                    key = f'{g}_{s}'
                    if key in summary:
                        for metric in records[g]:
                            val = summary[key].get(metric, None)
                            if val is not None:
                                records[g][metric].append(float(val))
                continue
            row = df.iloc[-1]
            for metric in records[g]:
                if metric in row:
                    records[g][metric].append(float(row[metric]))

    # Also pull from results_summary.json for completeness
    summary_path = os.path.join(DATA_DIR, 'results_summary.json')
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            results_summary = json.load(f)
    else:
        results_summary = {}

    # For groups where CSV might be missing, fill from summary
    for g in GROUPS:
        if len(records[g]['perplexity']) < 3:
            for s in SEEDS:
                key = f'{g}_{s}'
                if key in results_summary and len(records[g]['perplexity']) < 3:
                    for metric in records[g]:
                        val = results_summary[key].get(metric, None)
                        if val is not None:
                            records[g][metric].append(float(val))
        # Also check group-specific summary
        gsummary_path = os.path.join(DATA_DIR, f'group{g}_summary.json')
        if os.path.exists(gsummary_path) and len(records[g]['perplexity']) < 3:
            with open(gsummary_path) as f:
                gsummary = json.load(f)
            for s in SEEDS:
                key = f'{g}_{s}'
                if key in gsummary:
                    for metric in records[g]:
                        val = gsummary[key].get(metric, None)
                        if val is not None and len(records[g][metric]) < 3:
                            records[g][metric].append(float(val))

    # Print formatted table
    print(f"\n{'Group':<20} {'PPL':>14} {'ECE':>14} {'ConfErr':>14} {'HiConfFrac':>14}")
    print("-" * 78)
    table_rows = []
    for g in GROUPS:
        vals = records[g]
        if len(vals['perplexity']) == 0:
            print(f"  {g} ({GROUP_NAMES.get(g, g)}): NO DATA")
            continue
        row = {'group': g}
        for metric in ['perplexity', 'ece', 'conf_on_errors', 'high_conf_error_frac']:
            arr = np.array(vals[metric])
            row[f'{metric}_mean'] = np.mean(arr)
            row[f'{metric}_std'] = np.std(arr, ddof=1) if len(arr) > 1 else 0.0
        table_rows.append(row)
        print(f"  {g} ({GROUP_NAMES.get(g, g):<14}) "
              f"{row['perplexity_mean']:7.2f}+/-{row['perplexity_std']:4.2f}  "
              f"{row['ece_mean']:8.5f}+/-{row['ece_std']:6.5f}  "
              f"{row['conf_on_errors_mean']:6.4f}+/-{row['conf_on_errors_std']:6.4f}  "
              f"{row['high_conf_error_frac_mean']:7.5f}+/-{row['high_conf_error_frac_std']:7.5f}")

    # Statistical tests: paired comparisons
    comparisons = [('C', 'A'), ('C', 'B'), ('C', 'D'), ('B', 'A'), ('H', 'A')]
    print(f"\n{'Comparison':<12} {'Metric':<18} {'t-stat':>8} {'p-value':>10} {'Cohen d':>10}")
    print("-" * 62)
    stat_rows = []
    for g1, g2 in comparisons:
        if len(records[g1]['perplexity']) < 2 or len(records[g2]['perplexity']) < 2:
            print(f"  {g1} vs {g2}: insufficient data for t-test")
            continue
        for metric in ['perplexity', 'ece', 'conf_on_errors']:
            a = np.array(records[g1][metric])
            b = np.array(records[g2][metric])
            if len(a) < 2 or len(b) < 2:
                continue
            t_stat, p_val = stats.ttest_ind(a, b)
            d = cohens_d(a, b)
            print(f"  {g1} vs {g2}    {metric:<18} {t_stat:8.3f} {p_val:10.6f} {d:10.3f}")
            stat_rows.append({
                'comparison': f'{g1}_vs_{g2}', 'metric': metric,
                't_stat': t_stat, 'p_value': p_val, 'cohens_d': d,
            })

    # Save CSV
    csv_path = os.path.join(OUT_DIR, 'table_main_results.csv')
    if table_rows:
        pd.DataFrame(table_rows).to_csv(csv_path, index=False)
        print(f"\n  Saved: {csv_path}")

    stat_csv_path = os.path.join(OUT_DIR, 'table_stat_tests.csv')
    if stat_rows:
        pd.DataFrame(stat_rows).to_csv(stat_csv_path, index=False)
        print(f"  Saved: {stat_csv_path}")

    return records


# ================================================================
# 3B: Curvature Analysis
# ================================================================

def section_3b():
    print("\n" + "=" * 70)
    print("3B: Curvature Analysis")
    print("=" * 70)

    # --- Plot 1: Mean curvature vs training step, per group ---
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {'A': '#1f77b4', 'B': '#ff7f0e', 'C': '#2ca02c', 'D': '#d62728', 'H': '#9467bd'}
    for g in GROUPS:
        all_steps = None
        all_curves = []
        for s in SEEDS:
            df = load_geometry_csv(g, s)
            if df is None:
                continue
            # Mean curvature across layers per step
            mean_curv = df.groupby('step')['curvature'].mean()
            all_curves.append(mean_curv)
            if all_steps is None:
                all_steps = mean_curv.index.values
        if not all_curves:
            continue
        # Align on common steps
        common = all_curves[0].index
        for mc in all_curves[1:]:
            common = common.intersection(mc.index)
        mat = np.array([mc.loc[common].values for mc in all_curves])
        mean_line = mat.mean(axis=0)
        std_line = mat.std(axis=0) if mat.shape[0] > 1 else np.zeros_like(mean_line)
        ax.plot(common, mean_line, label=GROUP_NAMES.get(g, g), color=colors.get(g, 'gray'))
        ax.fill_between(common, mean_line - std_line, mean_line + std_line,
                        alpha=0.15, color=colors.get(g, 'gray'))

    ax.set_xlabel('Training Step')
    ax.set_ylabel('Mean Curvature (Frobenius)')
    ax.legend()
    fig.savefig(os.path.join(OUT_DIR, 'curvature_vs_step.pdf'))
    plt.close(fig)
    print("  Saved: curvature_vs_step.pdf")

    # --- Plot 2: Per-layer curvature at final step (bar chart) ---
    fig, ax = plt.subplots(figsize=(7, 4))
    n_layers = 6
    bar_width = 0.15
    x = np.arange(n_layers)
    for idx, g in enumerate(GROUPS):
        layer_curvs = []
        for s in SEEDS:
            df = load_geometry_csv(g, s)
            if df is None:
                continue
            last_step = df['step'].max()
            final = df[df['step'] == last_step]
            layer_curvs.append(final.sort_values('layer')['curvature'].values)
        if not layer_curvs:
            continue
        mat = np.array(layer_curvs)
        means = mat.mean(axis=0)
        stds = mat.std(axis=0) if mat.shape[0] > 1 else np.zeros_like(means)
        ax.bar(x + idx * bar_width, means, bar_width, yerr=stds,
               label=GROUP_NAMES.get(g, g), color=colors.get(g, 'gray'), capsize=2)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Curvature')
    ax.set_xticks(x + bar_width * (len(GROUPS) - 1) / 2)
    ax.set_xticklabels([str(i) for i in range(n_layers)])
    ax.legend()
    fig.savefig(os.path.join(OUT_DIR, 'curvature_per_layer_final.pdf'))
    plt.close(fig)
    print("  Saved: curvature_per_layer_final.pdf")

    # --- Component-level non-normality for Group A ---
    print("\n  Component-level non-normality (Group A, seed 42, final):")
    ckpt = load_checkpoint('A', 42, 'final')
    if ckpt is not None:
        model = TransformerLM(group='A').to('cpu')
        model.load_state_dict(ckpt['model'])
        model.eval()

        component_results = []
        for i, block in enumerate(model.blocks):
            attn = block.attn
            ffn = block.ffn

            components = {
                'W_Q': attn.W_q.weight.detach().float(),
                'W_K': attn.W_k.weight.detach().float(),
                'W_V': attn.W_v.weight.detach().float(),
                'W_O': attn.W_o.weight.detach().float(),
            }
            # FFN effective operator
            W1 = ffn.w1.weight.detach().float()  # (d_ff, d_model)
            W2 = ffn.w2.weight.detach().float()  # (d_model, d_ff)
            components['FFN(W2@W1)'] = W2 @ W1  # (d_model, d_model)

            for name, M in components.items():
                # For non-square matrices, use M^T M or M M^T (whichever is smaller)
                if M.shape[0] != M.shape[1]:
                    if M.shape[0] < M.shape[1]:
                        M = M @ M.T
                    else:
                        M = M.T @ M

                fro_sq = torch.sum(M ** 2).item()
                try:
                    eigs = torch.linalg.eigvals(M)
                    spec_sq = torch.sum(torch.abs(eigs) ** 2).item()
                    henrici = np.sqrt(max(fro_sq - spec_sq, 0)) / np.sqrt(max(fro_sq, 1e-12))

                    H_sym = (M + M.T) / 2
                    K_skew = (M - M.T) / 2
                    comm = H_sym @ K_skew - K_skew @ H_sym
                    curvature = torch.norm(comm, p='fro').item()
                except Exception as e:
                    henrici, curvature = 0.0, 0.0

                component_results.append({
                    'layer': i, 'component': name,
                    'henrici': henrici, 'curvature': curvature,
                })

        comp_df = pd.DataFrame(component_results)
        print(comp_df.to_string(index=False))

        # Summarize: which component contributes most
        summary = comp_df.groupby('component')[['henrici', 'curvature']].mean()
        print("\n  Mean across layers:")
        print(summary.to_string())

        comp_df.to_csv(os.path.join(OUT_DIR, 'component_nonnormality.csv'), index=False)
        print(f"\n  Saved: component_nonnormality.csv")

        highest = comp_df.loc[comp_df['curvature'].idxmax()]
        print(f"\n  Highest curvature: layer {highest['layer']}, "
              f"component {highest['component']}, curvature={highest['curvature']:.4f}")

        del model
        torch.cuda.empty_cache()


# ================================================================
# 3C: Beta Analysis (Groups B, C)
# ================================================================

def section_3c():
    print("\n" + "=" * 70)
    print("3C: Beta Analysis (Groups B, C)")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax_idx, g in enumerate(['B', 'C']):
        ax = axes[ax_idx]
        for s in SEEDS:
            ckpt = load_checkpoint(g, s, 'final')
            if ckpt is None:
                continue

            model = TransformerLM(group=g).to('cpu')
            model.load_state_dict(ckpt['model'])
            model.eval()

            # Extract learned alpha/beta by doing a dummy forward pass
            # OR read them from the geometry CSV last row
            # Better: read from geometry CSV for the last logged step
            df = load_geometry_csv(g, s)
            if df is not None:
                last_step = df['step'].max()
                final_geo = df[df['step'] == last_step].sort_values('layer')
                alphas = final_geo['alpha'].values
                betas = final_geo['beta'].values
                layers = final_geo['layer'].values

                ax.plot(layers, betas, 'o-', label=f'seed {s}', markersize=4)
                print(f"  Group {g}, seed {s}: beta = {betas}")

            del model

        ax.set_xlabel('Layer')
        if ax_idx == 0:
            ax.set_ylabel('$\\beta$ value')
        ax.set_title(f'Group {g}', fontsize=10)  # minimal title for context
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'beta_per_layer.pdf'))
    plt.close(fig)
    print("  Saved: beta_per_layer.pdf")

    # Also plot alpha
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax_idx, g in enumerate(['B', 'C']):
        ax = axes[ax_idx]
        for s in SEEDS:
            df = load_geometry_csv(g, s)
            if df is None:
                continue
            last_step = df['step'].max()
            final_geo = df[df['step'] == last_step].sort_values('layer')
            ax.plot(final_geo['layer'].values, final_geo['alpha'].values,
                    's-', label=f'seed {s}', markersize=4)
        ax.set_xlabel('Layer')
        if ax_idx == 0:
            ax.set_ylabel('$\\alpha$ value')
        ax.set_title(f'Group {g}', fontsize=10)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'alpha_per_layer.pdf'))
    plt.close(fig)
    print("  Saved: alpha_per_layer.pdf")


# ================================================================
# 3D: Pseudospectrum
# ================================================================

def section_3d():
    print("\n" + "=" * 70)
    print("3D: Pseudospectrum (Group A, seed 42, layer 3)")
    print("=" * 70)

    ckpt = load_checkpoint('A', 42, 'final')
    if ckpt is None:
        print("  Cannot load checkpoint. Skipping.")
        return

    model = TransformerLM(group='A').to('cpu')
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Extract layer 3 FFN weights
    block = model.blocks[3]
    W1 = block.ffn.w1.weight.detach().float().numpy()  # (1536, 384)
    W2 = block.ffn.w2.weight.detach().float().numpy()  # (384, 1536)
    M = W2 @ W1  # (384, 384)
    print(f"  M shape: {M.shape}, ||M||_F = {np.linalg.norm(M, 'fro'):.4f}")

    # Eigenvalues
    eigs = np.linalg.eigvals(M)
    print(f"  Eigenvalue range: real [{eigs.real.min():.3f}, {eigs.real.max():.3f}], "
          f"imag [{eigs.imag.min():.3f}, {eigs.imag.max():.3f}]")

    # Pseudospectrum: sigma_min(zI - M) on a grid
    # Use 100x100 grid centered on eigenvalue cloud
    margin = 1.5
    re_min, re_max = eigs.real.min() - margin, eigs.real.max() + margin
    im_min, im_max = eigs.imag.min() - margin, eigs.imag.max() + margin

    grid_n = 80
    re_vals = np.linspace(re_min, re_max, grid_n)
    im_vals = np.linspace(im_min, im_max, grid_n)
    RE, IM = np.meshgrid(re_vals, im_vals)

    print(f"  Computing pseudospectrum on {grid_n}x{grid_n} grid...")
    sigma_min_grid = np.zeros((grid_n, grid_n))

    from scipy.linalg import svdvals
    I_mat = np.eye(M.shape[0])
    for i in range(grid_n):
        for j in range(grid_n):
            z = complex(RE[i, j], IM[i, j])
            R = z * I_mat - M
            try:
                sigma_min_grid[i, j] = svdvals(R)[-1]
            except Exception:
                sigma_min_grid[i, j] = 0.0
        if (i + 1) % 20 == 0:
            print(f"    Row {i + 1}/{grid_n}", flush=True)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))

    # Log scale for contours
    log_sigma = np.log10(np.maximum(sigma_min_grid, 1e-10))
    levels = np.arange(-4, 2, 0.5)
    cs = ax.contourf(RE, IM, log_sigma, levels=levels, cmap='RdYlBu_r')
    ax.contour(RE, IM, log_sigma, levels=levels, colors='k', linewidths=0.3, alpha=0.5)
    cbar = fig.colorbar(cs, ax=ax)
    cbar.set_label('$\\log_{10} \\sigma_{\\min}(zI - M)$')

    # Eigenvalues
    ax.plot(eigs.real, eigs.imag, 'k.', markersize=2, alpha=0.6)

    ax.set_xlabel('Re($z$)')
    ax.set_ylabel('Im($z$)')
    ax.set_aspect('equal')

    fig.savefig(os.path.join(OUT_DIR, 'pseudospectrum_A_layer3.pdf'))
    plt.close(fig)
    print("  Saved: pseudospectrum_A_layer3.pdf")

    # Henrici for this matrix
    fro_sq = np.sum(M ** 2)
    spec_sq = np.sum(np.abs(eigs) ** 2)
    henrici = np.sqrt(max(fro_sq - spec_sq, 0)) / np.sqrt(max(fro_sq, 1e-12))
    print(f"  Henrici departure: {henrici:.4f}")

    del model
    torch.cuda.empty_cache()


# ================================================================
# 3E: Group H Summary Table
# ================================================================

def section_3e():
    print("\n" + "=" * 70)
    print("3E: Group H Summary")
    print("=" * 70)

    summary_path = os.path.join(DATA_DIR, 'groupH_summary.json')
    if not os.path.exists(summary_path):
        print("  groupH_summary.json not found. Skipping.")
        return

    with open(summary_path) as f:
        h_summary = json.load(f)

    # Collect H metrics
    h_ppl = []
    h_ece = []
    h_conf = []
    h_hcf = []
    h_disc_acc = []
    for s in SEEDS:
        key = f'H_{s}'
        if key not in h_summary:
            continue
        entry = h_summary[key]
        h_ppl.append(float(entry['perplexity']))
        h_ece.append(float(entry['ece']))
        h_conf.append(float(entry['conf_on_errors']))
        h_hcf.append(float(entry['high_conf_error_frac']))
        h_disc_acc.append(float(entry.get('disc_accuracy', 0)))

    print(f"  H: PPL={np.mean(h_ppl):.2f}+/-{np.std(h_ppl, ddof=1):.2f}, "
          f"ECE={np.mean(h_ece):.5f}+/-{np.std(h_ece, ddof=1):.5f}, "
          f"ConfErr={np.mean(h_conf):.4f}+/-{np.std(h_conf, ddof=1):.4f}")
    print(f"  Disc accuracy: {np.mean(h_disc_acc):.4f}")

    # Compare with Group A from results_summary
    a_ppl = []
    a_ece = []
    a_conf = []
    rs_path = os.path.join(DATA_DIR, 'results_summary.json')
    if os.path.exists(rs_path):
        with open(rs_path) as f:
            rs = json.load(f)
        for s in SEEDS:
            key = f'A_{s}'
            if key in rs:
                a_ppl.append(float(rs[key]['perplexity']))
                a_ece.append(float(rs[key]['ece']))
                a_conf.append(float(rs[key]['conf_on_errors']))

    if len(a_ppl) >= 2 and len(h_ppl) >= 2:
        print("\n  H vs A statistical comparison:")
        for name, h_arr, a_arr in [('perplexity', h_ppl, a_ppl),
                                     ('ece', h_ece, a_ece),
                                     ('conf_on_errors', h_conf, a_conf)]:
            t_stat, p_val = stats.ttest_ind(h_arr, a_arr)
            d = cohens_d(np.array(h_arr), np.array(a_arr))
            print(f"    {name:<18} t={t_stat:7.3f}  p={p_val:.6f}  d={d:.3f}")

    # Save
    h_df = pd.DataFrame([h_summary[f'H_{s}'] for s in SEEDS if f'H_{s}' in h_summary])
    h_df.to_csv(os.path.join(OUT_DIR, 'table_groupH.csv'), index=False)
    print(f"\n  Saved: table_groupH.csv")


# ================================================================
# Main
# ================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='3a', help='Start from section (3a/3b/3c/3d/3e)')
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 3: Post-Training Analysis")
    print(f"Device: {DEVICE}")
    print(f"Output: {OUT_DIR}")
    print(f"Starting from: {args.start}")
    print("=" * 70)

    sections = [('3a', section_3a), ('3b', section_3b), ('3c', section_3c),
                ('3d', section_3d), ('3e', section_3e)]
    started = False
    for name, fn in sections:
        if name == args.start:
            started = True
        if started:
            fn()

    print("\n" + "=" * 70)
    print("PHASE 3 COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
