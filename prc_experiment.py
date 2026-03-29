"""
GPU-Scale Experiment: Projection Residual Connections (PRC)
============================================================
Implements Section 8.9 of "The Geometric Origin of the Expressivity-Stability
Tradeoff in Deep Networks" by Zeta Hoi-Ho Cheng.

Experiment: CIFAR-10 classification with ResNet-20 style architecture.
Four groups:
  A: Standard residual  (y = x + F(x))
  B: PRC with beta >= 0 (constructive only, beta via sigmoid)
  C: PRC with beta in [-1,1] (full, beta via tanh)
  D: Forced normalization (weight matrices symmetrized)

Metrics tracked:
  - Train/test accuracy and loss
  - Henrici number per layer
  - Curvature ||[H,K]||_F per layer
  - Learned alpha/beta values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import json
import os
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# PRC Components
# ============================================================

class AttentionAlphaBeta(nn.Module):
    """Attention Alpha-Beta mechanism (Eq. 2 in the paper).
    Produces input-dependent alpha in [0,1] and beta in [-1,1] or [0,1].
    """
    def __init__(self, channels, allow_negative_beta=True):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        hidden = max(channels // 4, 4)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        self.allow_negative_beta = allow_negative_beta

    def forward(self, x):
        # x: (B, C, H, W)
        g = self.gap(x).view(x.size(0), -1)  # (B, C)
        h = F.relu(self.fc1(g))
        out = self.fc2(h)  # (B, 2)
        alpha = torch.sigmoid(out[:, 0])  # [0, 1]
        if self.allow_negative_beta:
            beta = torch.tanh(out[:, 1])  # [-1, 1]
        else:
            beta = torch.sigmoid(out[:, 1])  # [0, 1]
        return alpha, beta


class ProjectionPath(nn.Module):
    """Backward projection path: compress to k dims, re-expand.
    G(x) = W2 * sigma(W1 * x), implemented as 1x1 convolutions.
    """
    def __init__(self, channels, bottleneck_ratio=0.25):
        super().__init__()
        k = max(int(channels * bottleneck_ratio), 1)
        self.compress = nn.Conv2d(channels, k, 1, bias=False)
        self.expand = nn.Conv2d(k, channels, 1, bias=False)

    def forward(self, x):
        return self.expand(F.relu(self.compress(x)))


# ============================================================
# Residual Block Variants
# ============================================================

class BasicBlock(nn.Module):
    """Standard ResNet basic block with configurable residual type."""
    def __init__(self, in_channels, out_channels, stride=1, group='A'):
        super().__init__()
        self.group = group

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut for dimension mismatch
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        # PRC components (groups B and C)
        if group in ('B', 'C'):
            allow_neg = (group == 'C')
            self.alpha_beta = AttentionAlphaBeta(out_channels, allow_negative_beta=allow_neg)
            self.proj = ProjectionPath(out_channels)

        # For tracking
        self.last_alpha = None
        self.last_beta = None

    def _symmetrize_weights(self):
        """Force conv weights to be symmetric (group D)."""
        with torch.no_grad():
            w1 = self.conv1.weight.data
            # Symmetrize across spatial dims
            self.conv1.weight.data = (w1 + w1.flip(-1).flip(-2)) / 2
            w2 = self.conv2.weight.data
            self.conv2.weight.data = (w2 + w2.flip(-1).flip(-2)) / 2

    def forward(self, x):
        if self.group == 'D':
            self._symmetrize_weights()

        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        fx = self.bn2(self.conv2(out))

        if self.group == 'A' or self.group == 'D':
            # Standard: y = x + F(x)
            y = identity + fx
            self.last_alpha = 1.0
            self.last_beta = 0.0
        else:
            # PRC: y = x + alpha * F(x) + beta * G(x)
            alpha, beta = self.alpha_beta(identity)
            gx = self.proj(identity)
            # alpha, beta: (B,) -> (B, 1, 1, 1)
            a = alpha.view(-1, 1, 1, 1)
            b = beta.view(-1, 1, 1, 1)
            y = identity + a * fx + b * gx
            self.last_alpha = alpha.mean().item()
            self.last_beta = beta.mean().item()

        return F.relu(y)


# ============================================================
# ResNet-20 Style Model
# ============================================================

class ResNet20(nn.Module):
    def __init__(self, group='A', num_classes=10):
        super().__init__()
        self.group = group

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)

        # 3 stages, each with 3 blocks = 9 blocks total (roughly ResNet-20)
        self.layer1 = self._make_layer(16, 16, 3, 1, group)
        self.layer2 = self._make_layer(16, 32, 3, 2, group)
        self.layer3 = self._make_layer(32, 64, 3, 2, group)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, in_ch, out_ch, num_blocks, stride, group):
        layers = [BasicBlock(in_ch, out_ch, stride, group)]
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1, group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def get_blocks(self):
        blocks = []
        for layer in [self.layer1, self.layer2, self.layer3]:
            for block in layer:
                blocks.append(block)
        return blocks


# ============================================================
# Curvature and Non-normality Metrics
# ============================================================

def compute_curvature_metrics(model):
    """Compute Henrici number and curvature ||[H,K]||_F for each block's
    effective weight (conv2 weight reshaped as a 2D matrix)."""
    metrics = []
    for i, block in enumerate(model.get_blocks()):
        W = block.conv2.weight.data.view(block.conv2.weight.size(0), -1).cpu().numpy()

        # Henrici number: dF(T) = sqrt(||T||_F^2 - sum|lambda_i|^2) / ||T||_F
        fro_norm_sq = np.sum(W ** 2)
        try:
            eigenvalues = np.linalg.eigvals(W @ W.T)  # for non-square, use singular values
            eig_sum_sq = np.sum(np.abs(eigenvalues))
            # Use SVD-based approach for non-square matrices
            svs = np.linalg.svd(W, compute_uv=False)
            sv_sum_sq = np.sum(svs ** 2)

            # For non-square W, compute curvature of W^T W or W W^T
            M = W @ W.T if W.shape[0] <= W.shape[1] else W.T @ W
            H = (M + M.T) / 2
            K = (M - M.T) / 2
            commutator = H @ K - K @ H
            curvature = np.linalg.norm(commutator, 'fro')

            # Henrici on the square matrix
            eigs = np.linalg.eigvals(M)
            eig_sq_sum = np.sum(np.abs(eigs) ** 2)
            m_fro_sq = np.sum(M ** 2)
            if m_fro_sq > 1e-12:
                henrici = np.sqrt(max(m_fro_sq - eig_sq_sum, 0)) / np.sqrt(m_fro_sq)
            else:
                henrici = 0.0
        except Exception:
            henrici = 0.0
            curvature = 0.0

        metrics.append({
            'block': i,
            'henrici': float(henrici),
            'curvature': float(curvature),
            'alpha': block.last_alpha if block.last_alpha is not None else None,
            'beta': block.last_beta if block.last_beta is not None else None,
        })
    return metrics


# ============================================================
# Training and Evaluation
# ============================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    all_confs = []
    all_correct = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            loss = criterion(out, y)
            probs = F.softmax(out, dim=1)
            conf, pred = probs.max(1)
            total_loss += loss.item() * x.size(0)
            correct += (pred == y).sum().item()
            total += x.size(0)
            all_confs.extend(conf.cpu().numpy())
            all_correct.extend((pred == y).cpu().numpy())
    acc = correct / total
    avg_loss = total_loss / total

    # Expected Calibration Error (ECE)
    confs = np.array(all_confs)
    corrects = np.array(all_correct)
    n_bins = 15
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs > bin_boundaries[i]) & (confs <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = corrects[mask].mean()
            bin_conf = confs[mask].mean()
            ece += mask.sum() * abs(bin_acc - bin_conf)
    ece /= len(confs)

    return avg_loss, acc, ece


# ============================================================
# Main Experiment
# ============================================================

def run_experiment(group, seed, epochs=100, lr=0.1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Data
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)

    # Model
    model = ResNet20(group=group).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    group_names = {'A': 'Standard', 'B': 'PRC(b>=0)', 'C': 'PRC(full)', 'D': 'ForcedNormal'}
    print(f"\n{'='*60}")
    print(f"Group {group}: {group_names[group]} | Seed {seed} | Params: {param_count:,}")
    print(f"{'='*60}")

    history = []
    best_acc = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, trainloader, optimizer, criterion)
        test_loss, test_acc, ece = evaluate(model, testloader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        if test_acc > best_acc:
            best_acc = test_acc

        record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'test_loss': test_loss,
            'test_acc': test_acc,
            'ece': ece,
            'lr': optimizer.param_groups[0]['lr'],
            'time': elapsed,
        }

        # Compute curvature metrics every 10 epochs or at the end
        if epoch % 10 == 0 or epoch == epochs:
            # Run a dummy forward pass to get alpha/beta values
            model.eval()
            with torch.no_grad():
                dummy = next(iter(testloader))[0][:1].to(DEVICE)
                model(dummy)
            metrics = compute_curvature_metrics(model)
            record['curvature_metrics'] = metrics

            avg_henrici = np.mean([m['henrici'] for m in metrics])
            avg_curvature = np.mean([m['curvature'] for m in metrics])
            alphas = [m['alpha'] for m in metrics if m['alpha'] is not None]
            betas = [m['beta'] for m in metrics if m['beta'] is not None]

            if alphas:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train {train_acc:.4f} | Test {test_acc:.4f} | "
                      f"ECE {ece:.4f} | Henrici {avg_henrici:.4f} | "
                      f"Curv {avg_curvature:.2f} | "
                      f"a={np.mean(alphas):.3f} b={np.mean(betas):.3f} | "
                      f"{elapsed:.1f}s")
            else:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"Train {train_acc:.4f} | Test {test_acc:.4f} | "
                      f"ECE {ece:.4f} | Henrici {avg_henrici:.4f} | "
                      f"Curv {avg_curvature:.2f} | {elapsed:.1f}s")
        elif epoch % 5 == 0:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train {train_acc:.4f} | Test {test_acc:.4f} | "
                  f"ECE {ece:.4f} | {elapsed:.1f}s")

        history.append(record)

    return {
        'group': group,
        'group_name': group_names[group],
        'seed': seed,
        'param_count': param_count,
        'best_test_acc': best_acc,
        'final_test_acc': test_acc,
        'final_train_acc': train_acc,
        'final_ece': ece,
        'history': history,
    }


def main():
    print("=" * 60)
    print("PRC GPU-Scale Experiment: CIFAR-10 + ResNet-20")
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    groups = ['A', 'B', 'C', 'D']
    seeds = [42, 123, 456]  # 3 seeds for statistical analysis
    epochs = 100
    all_results = []

    for seed in seeds:
        for group in groups:
            result = run_experiment(group, seed, epochs=epochs)
            all_results.append(result)

            # Save intermediate results
            save_path = os.path.join(os.path.dirname(__file__), 'prc_results.json')
            # Convert history to serializable format
            serializable = []
            for r in all_results:
                sr = {k: v for k, v in r.items() if k != 'history'}
                sr['history_summary'] = {
                    'final_epoch': r['history'][-1] if r['history'] else {},
                    'curvature_at_end': r['history'][-1].get('curvature_metrics', []) if r['history'] else [],
                }
                serializable.append(sr)
            with open(save_path, 'w') as f:
                json.dump(serializable, f, indent=2, default=str)

    # ============================================================
    # Final Analysis
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    group_names = {'A': 'Standard', 'B': 'PRC(b>=0)', 'C': 'PRC(full)', 'D': 'ForcedNormal'}

    for group in groups:
        group_results = [r for r in all_results if r['group'] == group]
        test_accs = [r['best_test_acc'] for r in group_results]
        eces = [r['final_ece'] for r in group_results]
        mean_acc = np.mean(test_accs)
        std_acc = np.std(test_accs)
        mean_ece = np.mean(eces)

        print(f"\nGroup {group} ({group_names[group]}):")
        print(f"  Best Test Acc: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  ECE:           {mean_ece:.4f}")
        print(f"  Per-seed:      {[f'{a:.4f}' for a in test_accs]}")

    # Statistical tests (C vs others)
    from scipy import stats
    print("\n" + "-" * 40)
    print("Statistical Tests: Group C vs Others")
    print("-" * 40)

    c_accs = [r['best_test_acc'] for r in all_results if r['group'] == 'C']
    for group in ['A', 'B', 'D']:
        other_accs = [r['best_test_acc'] for r in all_results if r['group'] == group]
        t_stat, p_val = stats.ttest_rel(c_accs, other_accs)
        diff = np.mean(c_accs) - np.mean(other_accs)
        pooled_std = np.sqrt((np.std(c_accs)**2 + np.std(other_accs)**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0
        print(f"  C vs {group}: diff={diff:+.4f}, t={t_stat:.3f}, p={p_val:.4f}, d={cohens_d:.2f}")

    # Curvature analysis
    print("\n" + "-" * 40)
    print("Curvature Analysis (Final Epoch)")
    print("-" * 40)

    for group in groups:
        group_results = [r for r in all_results if r['group'] == group]
        all_curvatures = []
        all_henricis = []
        for r in group_results:
            last = r['history'][-1]
            if 'curvature_metrics' in last:
                for m in last['curvature_metrics']:
                    all_curvatures.append(m['curvature'])
                    all_henricis.append(m['henrici'])
        if all_curvatures:
            print(f"  Group {group}: Curvature={np.mean(all_curvatures):.2f}±{np.std(all_curvatures):.2f}, "
                  f"Henrici={np.mean(all_henricis):.4f}±{np.std(all_henricis):.4f}")


if __name__ == '__main__':
    main()
