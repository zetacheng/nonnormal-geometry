"""Phase 0: Environment setup and sanity check."""
import torch
import torch.nn as nn
import math

print("=" * 60)
print("PHASE 0: SANITY CHECK")
print("=" * 60)

# 1. GPU verification
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# 2. Quick Transformer memory test
print("\n--- Transformer Memory Test ---")

class MiniTransformer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, n_layers=12,
                 vocab_size=50257, max_seq=512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.layers = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)
        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.layers(h, mask=mask, is_causal=True)
        h = self.norm(h)
        return self.head(h)

working_bs = None
for bs in [16, 8, 4]:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    try:
        model = MiniTransformer().cuda().half()
        x = torch.randint(0, 50257, (bs, 512), device='cuda')
        with torch.cuda.amp.autocast():
            out = model(x)
            loss = out.view(-1, 50257).float().mean()
        loss.backward()
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  batch_size={bs}: PASS (peak VRAM: {peak:.2f} GB)")
        working_bs = bs
        del model, x, out, loss
        torch.cuda.empty_cache()
        break
    except torch.cuda.OutOfMemoryError:
        print(f"  batch_size={bs}: OOM")
        del model
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  batch_size={bs}: ERROR - {e}")
        try:
            del model
        except:
            pass
        torch.cuda.empty_cache()

print(f"  Working batch size: {working_bs}")

# 3. Test geometric measurement functions
print("\n--- Geometric Measurement Test ---")
torch.manual_seed(42)
W = torch.randn(512, 512)

# Curvature
H = (W + W.T) / 2
K = (W - W.T) / 2
curvature = torch.norm(H @ K - K @ H, p='fro').item()
print(f"  Curvature ||[H,K]||_F = {curvature:.4f}")

# Henrici number
eigs = torch.linalg.eigvals(W)
fro_sq = torch.sum(W ** 2).item()
spec_sq = torch.sum(torch.abs(eigs) ** 2).item()
henrici = math.sqrt(max(fro_sq - spec_sq, 0)) / math.sqrt(fro_sq)
print(f"  Henrici number = {henrici:.4f}")

# Excess noise
excess_noise = fro_sq - spec_sq
print(f"  Excess noise S(T) = {excess_noise:.4f}")

# Spectral radius
spec_radius = torch.max(torch.abs(eigs)).item()
print(f"  Spectral radius = {spec_radius:.4f}")

# Eigenvalue computation time
import time
t0 = time.time()
for _ in range(12):
    _ = torch.linalg.eigvals(torch.randn(512, 512))
t1 = time.time()
print(f"  12x eigvals(512x512): {(t1-t0)*1000:.1f} ms ({(t1-t0)/12*1000:.1f} ms each)")

print("\n" + "=" * 60)
print(f"PHASE 0 COMPLETE")
print(f"  Working batch size: {working_bs}")
print(f"  Geometry functions: OK")
print("=" * 60)
