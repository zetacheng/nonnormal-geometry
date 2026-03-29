"""
Phase 2: Language Model Training with Geometry Logging
Decoder-only Transformer on WikiText-103, 4 residual variants.
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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiments', 'phase2_lm')
os.makedirs(os.path.join(BASE_DIR, 'checkpoints'), exist_ok=True)


# ============================================================
# RMSNorm
# ============================================================

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================
# Rotary Position Embedding (RoPE)
# ============================================================

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq = max_seq

    def forward(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    return x * cos + rotate_half(x) * sin


# ============================================================
# Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.d_head)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        q = self.W_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        cos, sin = self.rope(T)
        cos = cos[:T].unsqueeze(0).unsqueeze(0)
        sin = sin[:T].unsqueeze(0).unsqueeze(0)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Attention
        scale = math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale
        if mask is not None:
            scores = scores.masked_fill(mask[:T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


# ============================================================
# FFN Variants
# ============================================================

class StandardFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.gelu(self.w1(x))))


class ForcedNormalFFN(nn.Module):
    """FFN with symmetrized weight matrices."""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Symmetrize w1 (d_ff x d_model): use W @ W^T structure won't work for non-square
        # Instead: use (W + pad(W.T)) / 2 isn't valid for non-square
        # For non-square: apply SVD soft constraint: U S V^T -> V S V^T (makes it symmetric-like)
        # Simpler: just use the weight as-is but add regularization in loss
        # Actually for forced normality in the forward pass, we make the effective
        # Jacobian more normal by using tied weights: w2 = w1.T
        h = F.gelu(x @ self.w1.weight.T)  # (B, T, d_ff)
        return self.dropout(h @ self.w1.weight)   # tied: w2 = w1.T


# ============================================================
# PRC Components (for LM)
# ============================================================

class AttentionAlphaBetaLM(nn.Module):
    def __init__(self, d_model, allow_negative_beta=True):
        super().__init__()
        hidden = max(d_model // 32, 8)
        self.fc1 = nn.Linear(d_model, hidden, bias=False)
        self.fc2 = nn.Linear(hidden, 2, bias=False)
        self.allow_negative_beta = allow_negative_beta

    def forward(self, x):
        # x: (B, T, D) -> global average over T
        g = x.mean(dim=1)  # (B, D)
        h = F.relu(self.fc1(g))
        out = self.fc2(h)  # (B, 2)
        alpha = torch.sigmoid(out[:, 0])
        if self.allow_negative_beta:
            beta = torch.tanh(out[:, 1])
        else:
            beta = torch.sigmoid(out[:, 1])
        return alpha, beta  # (B,), (B,)


class ProjectionPathLM(nn.Module):
    def __init__(self, d_model, bottleneck=96):
        super().__init__()
        self.down = nn.Linear(d_model, bottleneck, bias=False)
        self.up = nn.Linear(bottleneck, d_model, bias=False)

    def forward(self, x):
        return self.up(F.relu(self.down(x)))


# ============================================================
# Transformer Block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, group='A'):
        super().__init__()
        self.group = group
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)

        if group == 'D':
            self.ffn = ForcedNormalFFN(d_model, d_ff, dropout)
        else:
            self.ffn = StandardFFN(d_model, d_ff, dropout)

        if group in ('B', 'C'):
            self.alpha_beta = AttentionAlphaBetaLM(d_model, allow_negative_beta=(group == 'C'))
            self.proj = ProjectionPathLM(d_model, bottleneck=d_model // 4)

        self.last_alpha = None
        self.last_beta = None

    def forward(self, x, mask=None):
        # Attention (always standard residual)
        x = x + self.attn(self.norm1(x), mask=mask)

        # FFN residual
        normed = self.norm2(x)
        ffn_out = self.ffn(normed)

        if self.group in ('A', 'D'):
            x = x + ffn_out
            self.last_alpha = 1.0
            self.last_beta = 0.0
        else:
            alpha, beta = self.alpha_beta(normed)
            proj_out = self.proj(normed)
            a = alpha.unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
            b = beta.unsqueeze(1).unsqueeze(2)
            x = x + a * ffn_out + b * proj_out
            self.last_alpha = alpha.mean().item()
            self.last_beta = beta.mean().item()

        return x


# ============================================================
# Full Language Model
# ============================================================

class TransformerLM(nn.Module):
    def __init__(self, vocab_size=50257, d_model=384, n_heads=6, d_ff=1536,
                 n_layers=6, max_seq=256, dropout=0.1, group='A'):
        super().__init__()
        self.d_model = d_model
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout, group)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.head.weight = self.tok_embed.weight

        # Causal mask
        mask = torch.tril(torch.ones(max_seq, max_seq))
        self.register_buffer('mask', mask)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_embed(idx) * math.sqrt(self.d_model)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, self.mask)
        x = self.norm(x)
        return self.head(x)

    def get_ffn_weights(self):
        """Get FFN first layer weights for geometry measurement."""
        weights = []
        for block in self.blocks:
            if hasattr(block.ffn, 'w1'):
                weights.append(block.ffn.w1.weight)
            else:
                weights.append(None)
        return weights


# ============================================================
# Geometry Measurement
# ============================================================

def measure_geometry(model):
    """Compute geometric metrics for all layers.
    Use W2 @ W1 as the effective FFN operator (d_model x d_model square matrix).
    This is the correct proxy for the FFN Jacobian's non-normality.
    """
    results = []
    for i, block in enumerate(model.blocks):
        # Get both FFN weights to form effective operator W2 @ W1
        if hasattr(block.ffn, 'w1') and hasattr(block.ffn, 'w2'):
            W1 = block.ffn.w1.weight.detach().float().cpu()  # (d_ff, d_model)
            W2 = block.ffn.w2.weight.detach().float().cpu()  # (d_model, d_ff)
            M = W2 @ W1  # (d_model, d_model) - square, generically non-normal
        elif hasattr(block.ffn, 'w1'):
            # ForcedNormal: tied weights, W2 = W1.T, so M = W1.T @ W1 (symmetric=normal)
            W1 = block.ffn.w1.weight.detach().float().cpu()
            M = W1.T @ W1
        else:
            continue

        fro_sq = torch.sum(M ** 2).item()
        try:
            eigs = torch.linalg.eigvals(M)
            spec_sq = torch.sum(torch.abs(eigs) ** 2).item()
            henrici = np.sqrt(max(fro_sq - spec_sq, 0)) / np.sqrt(max(fro_sq, 1e-12))

            H = (M + M.T) / 2
            K = (M - M.T) / 2
            comm = H @ K - K @ H
            curvature = torch.norm(comm, p='fro').item()

            excess_noise = max(fro_sq - spec_sq, 0)
            frobenius = torch.norm(M, p='fro').item()
        except Exception as e:
            print(f"  [WARN] Geometry error layer {i}: {e}")
            henrici, curvature, excess_noise, frobenius = 0, 0, 0, 0

        results.append({
            'layer': i,
            'henrici': henrici,
            'curvature': curvature,
            'excess_noise': excess_noise,
            'frobenius': frobenius,
            'alpha': block.last_alpha,
            'beta': block.last_beta,
        })
    return results


# ============================================================
# Data Loading (WikiText-103 via HuggingFace)
# ============================================================

def load_data(ctx_len=512):
    """Load WikiText-103 and tokenize with tiktoken."""
    import tiktoken
    from datasets import load_dataset

    print("Loading WikiText-103...")
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1')
    enc = tiktoken.get_encoding('gpt2')

    def tokenize_split(split_name):
        texts = ds[split_name]['text']
        all_tokens = []
        for t in texts:
            if t.strip():
                all_tokens.extend(enc.encode(t))
        tokens = torch.tensor(all_tokens, dtype=torch.long)
        # Chunk into sequences
        n_seq = len(tokens) // ctx_len
        tokens = tokens[:n_seq * ctx_len].view(n_seq, ctx_len)
        return tokens

    train_data = tokenize_split('train')
    val_data = tokenize_split('validation')

    print(f"  Train: {train_data.shape[0]} sequences of length {ctx_len}")
    print(f"  Val: {val_data.shape[0]} sequences of length {ctx_len}")
    return train_data, val_data


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(model, val_data, max_batches=50, batch_size=16):
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
            logits = model(inp)
        logits_f = logits.float()
        loss = F.cross_entropy(logits_f.view(-1, logits_f.size(-1)), tgt.reshape(-1))
        total_loss += loss.item() * tgt.numel()
        total_tokens += tgt.numel()

        # Confidence analysis
        probs = F.softmax(logits_f, dim=-1)
        conf, pred = probs.max(dim=-1)
        wrong = (pred != tgt)
        if wrong.any():
            all_wrong_confs.extend(conf[wrong].cpu().numpy())

        # ECE bins
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

    # ECE
    ece = 0
    for b in range(10):
        if all_bins[b] > 0:
            acc_b = all_bin_correct[b] / all_bins[b]
            conf_b = all_bin_conf[b] / all_bins[b]
            ece += all_bins[b] * abs(acc_b - conf_b)
    ece /= all_bins.sum()

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
# Training Loop
# ============================================================

def train_one_config(group, seed, train_data, val_data, total_steps=20000,
                     batch_size=32, ctx_len=256, lr=3e-4, warmup_steps=1000):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    group_names = {'A': 'Standard', 'B': 'PRC(b>=0)', 'C': 'PRC(full)', 'D': 'ForcedNormal'}
    print(f"\n{'='*60}")
    print(f"Group {group}: {group_names[group]} | Seed {seed}")
    print(f"{'='*60}")

    model = TransformerLM(group=group).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {param_count:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)
    scaler = GradScaler('cuda')

    # LR schedule: linear warmup then cosine decay
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Logging
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

        for i in range(0, n_train - batch_size, batch_size):
            if step >= total_steps:
                break

            idx = perm[i:i + batch_size]
            x = train_data[idx].to(DEVICE)
            inp = x[:, :-1]
            tgt = x[:, 1:]

            model.train()
            optimizer.zero_grad()

            with autocast('cuda'):
                logits = model(inp)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt.reshape(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += loss.item()
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

                metrics = evaluate(model, val_data, max_batches=50, batch_size=batch_size)
                metrics_csv.writerow([step, seed, group, metrics['perplexity'],
                                      metrics['loss'], metrics['ece'],
                                      metrics['conf_on_errors'], metrics['high_conf_error_frac']])
                metrics_writer.flush()

                elapsed = time.time() - t_start
                steps_per_sec = step / elapsed
                eta = (total_steps - step) / steps_per_sec

                # Get alpha/beta from last block
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

    # Final evaluation
    final_metrics = evaluate(model, val_data, max_batches=100, batch_size=batch_size)
    print(f"\n  FINAL: PPL={final_metrics['perplexity']:.1f} | "
          f"ECE={final_metrics['ece']:.4f} | "
          f"ConfErr={final_metrics['conf_on_errors']:.3f}")

    # Save final checkpoint
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

    return final_metrics


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("PHASE 2: Language Model Training with Geometry Logging")
    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0)})")
    print("=" * 60)

    train_data, val_data = load_data(ctx_len=256)

    groups = ['A', 'B', 'C', 'D']
    seeds = [42, 123, 456]
    total_steps = 20000

    all_results = {}

    for group in groups:
        for seed in seeds:
            key = f"{group}_{seed}"
            result = train_one_config(
                group=group, seed=seed,
                train_data=train_data, val_data=val_data,
                total_steps=total_steps,
            )
            all_results[key] = result

            # Save intermediate summary
            summary_path = os.path.join(BASE_DIR, 'results_summary.json')
            with open(summary_path, 'w') as f:
                json.dump({k: v for k, v in all_results.items()}, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)

    group_names = {'A': 'Standard', 'B': 'PRC(b>=0)', 'C': 'PRC(full)', 'D': 'ForcedNormal'}
    for group in groups:
        ppls = [all_results[f"{group}_{s}"]['perplexity'] for s in seeds]
        eces = [all_results[f"{group}_{s}"]['ece'] for s in seeds]
        print(f"  Group {group} ({group_names[group]}): "
              f"PPL={np.mean(ppls):.1f}+/-{np.std(ppls):.1f} | "
              f"ECE={np.mean(eces):.4f}")


if __name__ == '__main__':
    main()
