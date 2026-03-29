"""
Phase 1: RNN Non-normality Experiment (NNOG Paper)
Train a simple RNN on a delayed memory task and track non-normality growth.
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import os
import time

plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'serif'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'experiments', 'phase1_rnn')
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# RNN Model
# ============================================================

class SimpleRNN(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W_h = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.W_x = nn.Parameter(torch.randn(hidden_dim, input_dim) * 0.1)
        self.W_y = nn.Parameter(torch.randn(output_dim, hidden_dim) * 0.1)

    def forward(self, x_seq):
        # x_seq: (B, T, 1)
        B, T, _ = x_seq.shape
        h = torch.zeros(B, self.hidden_dim, device=x_seq.device)
        outputs = []
        for t in range(T):
            h = torch.tanh(h @ self.W_h.T + x_seq[:, t] @ self.W_x.T)
            y = h @ self.W_y.T
            outputs.append(y)
        return torch.stack(outputs, dim=1)  # (B, T, 1)


# ============================================================
# Geometric Measurements
# ============================================================

def compute_geometry(W):
    """Compute geometric metrics for a square matrix W."""
    W_cpu = W.detach().cpu().float()

    # Henrici number
    fro_sq = torch.sum(W_cpu ** 2).item()
    eigs = torch.linalg.eigvals(W_cpu)
    spec_sq = torch.sum(torch.abs(eigs) ** 2).item()
    henrici = np.sqrt(max(fro_sq - spec_sq, 0)) / np.sqrt(max(fro_sq, 1e-12))

    # Curvature
    H = (W_cpu + W_cpu.T) / 2
    K = (W_cpu - W_cpu.T) / 2
    comm = H @ K - K @ H
    curvature = torch.norm(comm, p='fro').item()

    # Spectral radius
    spec_radius = torch.max(torch.abs(eigs)).item()

    return {
        'henrici': henrici,
        'curvature': curvature,
        'spectral_radius': spec_radius,
        'eigenvalues': eigs.numpy(),
    }


# ============================================================
# Data Generation
# ============================================================

def generate_delayed_data(n_samples, seq_len=20, delay=5, noise_std=0.5):
    """Generate delayed identity mapping task data."""
    x = torch.randn(n_samples, seq_len, 1) * noise_std
    y = torch.zeros_like(x)
    y[:, delay:, :] = x[:, :-delay, :]
    return x, y


# ============================================================
# Pseudospectrum Computation
# ============================================================

def compute_pseudospectrum(W, grid_size=200, margin=1.0):
    """Compute pseudospectrum on a grid in the complex plane."""
    W_np = W.detach().cpu().float().numpy()
    eigs = np.linalg.eigvals(W_np)

    # Define grid around eigenvalues
    re_min, re_max = eigs.real.min() - margin, eigs.real.max() + margin
    im_min, im_max = eigs.imag.min() - margin, eigs.imag.max() + margin

    re_range = np.linspace(re_min, re_max, grid_size)
    im_range = np.linspace(im_min, im_max, grid_size)
    RE, IM = np.meshgrid(re_range, im_range)

    sigma_min_grid = np.zeros_like(RE)
    I = np.eye(W_np.shape[0])

    for i in range(grid_size):
        for j in range(grid_size):
            z = complex(RE[i, j], IM[i, j])
            M = z * I - W_np
            sigma_min_grid[i, j] = np.linalg.svd(M, compute_uv=False)[-1]

    return RE, IM, sigma_min_grid, eigs


# ============================================================
# Training
# ============================================================

def train():
    print("=" * 60)
    print("PHASE 1: RNN Non-normality Experiment")
    print("=" * 60)

    torch.manual_seed(42)
    np.random.seed(42)

    # Data
    seq_len, delay = 20, 5
    x_train, y_train = generate_delayed_data(2000, seq_len, delay)
    x_train, y_train = x_train.to(DEVICE), y_train.to(DEVICE)

    # Model
    model = SimpleRNN(input_dim=1, hidden_dim=64, output_dim=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 500
    batch_size = 64
    n_batches = len(x_train) // batch_size

    # Logging
    log = []
    eigs_epoch0 = None
    eigs_final = None

    print(f"\nTraining: {epochs} epochs, batch_size={batch_size}, delay={delay}")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(x_train), device=DEVICE)
        epoch_loss = 0

        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            xb, yb = x_train[idx], y_train[idx]

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred[:, delay:, :], yb[:, delay:, :])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= n_batches

        # Measure geometry every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            with torch.no_grad():
                geo = compute_geometry(model.W_h)

            record = {
                'epoch': epoch,
                'loss': epoch_loss,
                'henrici': geo['henrici'],
                'curvature': geo['curvature'],
                'spectral_radius': geo['spectral_radius'],
            }
            log.append(record)

            if epoch == 1:
                eigs_epoch0 = geo['eigenvalues']
            eigs_final = geo['eigenvalues']

            if epoch % 50 == 0 or epoch == 1:
                print(f"  Epoch {epoch:3d} | Loss {epoch_loss:.6f} | "
                      f"Henrici {geo['henrici']:.4f} | "
                      f"Curv {geo['curvature']:.2f} | "
                      f"rho {geo['spectral_radius']:.3f}")

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed:.1f}s")

    # Save CSV
    csv_path = os.path.join(OUT_DIR, 'rnn_training_log.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'loss', 'henrici', 'curvature', 'spectral_radius'])
        writer.writeheader()
        writer.writerows(log)
    print(f"Saved: {csv_path}")

    # Save eigenvalues
    np.save(os.path.join(OUT_DIR, 'eigenvalues_epoch0.npy'), eigs_epoch0)
    np.save(os.path.join(OUT_DIR, 'eigenvalues_final.npy'), eigs_final)

    # ============================================================
    # Figure 1: Learning Requires Non-normality (dual y-axis)
    # ============================================================
    epochs_arr = [r['epoch'] for r in log]
    losses = [r['loss'] for r in log]
    henricis = [r['henrici'] for r in log]

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss (MSE)', color='tab:red')
    ax1.plot(epochs_arr, losses, color='tab:red', linewidth=1.5, label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Henrici Index', color='tab:blue')
    ax2.plot(epochs_arr, henricis, color='tab:blue', linewidth=1.5, label='Henrici')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_learning_nonnormality.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig_learning_nonnormality.pdf")

    # ============================================================
    # Figure 2: Pseudospectrum (epoch 0 vs final)
    # ============================================================
    print("\nComputing pseudospectra (this may take a few minutes)...")

    # We need W_h at epoch 0 — re-init and compute
    torch.manual_seed(42)
    model_init = SimpleRNN(input_dim=1, hidden_dim=64, output_dim=1)
    W_init = model_init.W_h.data

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, W, title, eigs_arr in [
        (axes[0], W_init, 'Initialization', eigs_epoch0),
        (axes[1], model.W_h.data, 'After Training', eigs_final),
    ]:
        RE, IM, sigma_min, eigs_plot = compute_pseudospectrum(W, grid_size=150, margin=0.5)
        eps_levels = [0.1, 0.2, 0.5]
        colors = ['#2166ac', '#67a9cf', '#d1e5f0']
        for eps, color in zip(eps_levels, colors):
            ax.contour(RE, IM, sigma_min, levels=[eps], colors=[color], linewidths=1.2)
        ax.contourf(RE, IM, sigma_min, levels=[0, eps_levels[0]], colors=['#f4a582'], alpha=0.3)
        ax.plot(eigs_plot.real, eigs_plot.imag, 'r.', markersize=4, label='Eigenvalues')
        ax.set_xlabel('Re')
        ax.set_ylabel('Im')
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=9)
        # Add epsilon labels
        for eps in eps_levels:
            ax.text(0.02, 0.98 - eps_levels.index(eps) * 0.06,
                    f'eps={eps}', transform=ax.transAxes, fontsize=8,
                    va='top', color=colors[eps_levels.index(eps)])

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_pseudospectrum.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig_pseudospectrum.pdf")

    # ============================================================
    # Figure 3: Curvature evolution
    # ============================================================
    curvatures = [r['curvature'] for r in log]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(epochs_arr, curvatures, color='tab:green', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Curvature $\\|[H, K]\\|_F$')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, 'fig_curvature_evolution.pdf'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved: fig_curvature_evolution.pdf")

    # Summary
    print("\n" + "=" * 60)
    print("PHASE 1 SUMMARY")
    print("=" * 60)
    print(f"  Initial Henrici: {log[0]['henrici']:.4f} -> Final: {log[-1]['henrici']:.4f}")
    print(f"  Initial Curvature: {log[0]['curvature']:.4f} -> Final: {log[-1]['curvature']:.4f}")
    print(f"  Initial Loss: {log[0]['loss']:.6f} -> Final: {log[-1]['loss']:.6f}")
    print(f"  Initial Spec Radius: {log[0]['spectral_radius']:.4f} -> Final: {log[-1]['spectral_radius']:.4f}")
    increased = log[-1]['henrici'] > log[0]['henrici']
    print(f"  Henrici increased during training: {increased}")
    print(f"  Output dir: {OUT_DIR}")


if __name__ == '__main__':
    train()
