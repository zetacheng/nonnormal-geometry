# ============================================================
# COMBINED EXPERIMENT PLAN: PRC + NNOG
# Language Model + Geometric Measurements
# Hardware: 4060Ti 8GB VRAM, CUDA 12.6
# ============================================================
#
# This plan produces data for TWO papers simultaneously:
#   Paper 1 (PRC): Expressivity-stability tradeoff, destructive residuals
#   Paper 2 (NNOG): Non-normal operator geometry of deep networks
#
# TOTAL ESTIMATED TIME: ~4-5 days (can run overnight)
#   Phase 0: Setup + sanity check (~15 min)
#   Phase 1: RNN experiment for NNOG (~30 min)
#   Phase 2: LM training with geometry logging (~3-4 days)
#   Phase 3: Post-training analysis (~2-3 hours)
#   Phase 4: Hallucination / noise injection (~1 hour)

# ============================================================
# PHASE 0: ENVIRONMENT SETUP AND SANITY CHECK
# ============================================================

PHASE_0 = """
Before any experiments, set up the environment and verify everything works.

1. Install dependencies:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
   pip install scipy numpy matplotlib pandas tiktoken datasets

2. Verify GPU:
   - Print torch.cuda.is_available(), torch.cuda.get_device_name(0)
   - Print available VRAM: torch.cuda.get_device_properties(0).total_mem

3. Quick memory test:
   - Create a dummy 12-layer Transformer decoder (d_model=512, d_ff=2048, 
     8 heads, context=512, batch_size=16) in fp16
   - Run one forward + backward pass
   - Print peak VRAM usage: torch.cuda.max_memory_allocated()
   - If OOM: reduce batch_size to 8, retry
   - Report the working batch_size

4. Test geometric measurement functions:
   W = torch.randn(512, 512)
   H = (W + W.T) / 2
   K = (W - W.T) / 2
   curvature = torch.norm(H @ K - K @ H, p='fro').item()
   eigs = torch.linalg.eigvals(W)
   henrici = (torch.sum(W**2) - torch.sum(torch.abs(eigs)**2)).sqrt() / torch.norm(W, p='fro')
   Print both values to confirm they work.

Report the working configuration (batch_size, VRAM usage) before proceeding.
"""

# ============================================================
# PHASE 1: RNN NON-NORMALITY EXPERIMENT (FOR NNOG PAPER)
# ============================================================

PHASE_1 = """
Train a simple RNN on a delayed memory task and track non-normality growth.
This is for the NNOG paper (Non-normal Operator Geometry).

Task: Delayed identity mapping
  Input: x_t ~ N(0, 0.5²), sequence length = 20
  Target: y_t = x_{t-τ} for τ = 5 (5-step delay)
  This requires the RNN to store and retrieve information across time steps.

Model: Simple RNN (no bias, no input gate)
  h_t = tanh(W_h @ h_{t-1} + W_x @ x_t)
  y_t = W_y @ h_t
  Hidden dimension: d = 64

Training:
  - Optimizer: Adam, lr=1e-3
  - Loss: MSE on y_t vs target
  - Training samples: 2000 sequences
  - Epochs: 500
  - Batch size: 64

CRITICAL MEASUREMENTS (every 5 epochs):
  For the recurrent weight matrix W_h, compute and log:
  
  1. Henrici number:
     fro_sq = torch.sum(W_h ** 2)
     eigs = torch.linalg.eigvals(W_h)
     spec_sq = torch.sum(torch.abs(eigs) ** 2)
     henrici = torch.sqrt(torch.clamp(fro_sq - spec_sq, min=0)) / torch.sqrt(fro_sq)
  
  2. Curvature:
     H = (W_h + W_h.T) / 2
     K = (W_h - W_h.T) / 2
     curvature = torch.norm(H @ K - K @ H, p='fro')
  
  3. Spectral radius: max(|eigenvalues|)
  
  4. Training loss
  
  5. Eigenvalues of W_h (save full complex eigenvalue array for pseudospectrum plotting)

After training, generate:
  - Figure 1: Dual y-axis plot. Left axis: training loss (red). Right axis: Henrici index (blue).
    X-axis: epoch. Title: "Learning Requires Non-normality"
    Expected: as loss decreases, Henrici increases.
    
  - Figure 2: Pseudospectrum of W_h at epoch 0 vs epoch 500.
    For each: plot eigenvalues as red dots, and the ε-pseudospectrum boundary
    for ε = 0.1, 0.2, 0.5 as contour lines.
    To compute pseudospectrum: on a 200×200 grid in the complex plane around 
    the eigenvalues, compute σ_min(zI - W_h) at each grid point z.
    The ε-pseudospectrum boundary is the contour where σ_min = ε.
    
  - Figure 3: Curvature vs epoch (should increase or show interesting dynamics)

Save all per-epoch measurements as CSV: epoch, loss, henrici, curvature, spectral_radius
Save eigenvalues at epoch 0 and final epoch as separate numpy files.
Save all figures as PDF (300 dpi, no title, proper axis labels, suitable for LaTeX).
"""

# ============================================================
# PHASE 2: LANGUAGE MODEL TRAINING WITH GEOMETRY LOGGING
# ============================================================

PHASE_2 = """
Train a small Transformer language model on WikiText-103 with 4 residual variants.
Log geometric measurements during training for both PRC and NNOG papers.

DATA:
  - Dataset: WikiText-103 (use datasets library from HuggingFace)
  - Tokenizer: tiktoken "gpt2" encoding (50257 vocab)
  - Context length: 512 tokens
  - Split: use standard train/valid/test

MODEL: Decoder-only Transformer
  - Layers: 12
  - d_model: 512
  - d_ff: 2048 (4x d_model)
  - Attention heads: 8
  - Dropout: 0.1
  - Positional encoding: RoPE (rotary position embedding)
  - Normalization: RMSNorm (Pre-Norm style)
  - Use fp16 mixed precision (torch.cuda.amp) to fit in 8GB

4 TRAINING GROUPS:

Group A (Standard): Normal Pre-Norm Transformer
  Each block: h = h + Attn(RMSNorm(h)); h = h + FFN(RMSNorm(h))

Group B (PRC positive): PRC on FFN residual, β via sigmoid (β≥0)
  Each block: h = h + Attn(RMSNorm(h))  [attention untouched]
  FFN residual: h = h + α·FFN(RMSNorm(h)) + β·G(RMSNorm(h))
  where G(x) = Linear(512→128) → ReLU → Linear(128→512)  [bottleneck k=d/4]
  [α, β] = Attention_AlphaBeta(GAP(x))
  Attention_AlphaBeta: Linear(512→16) → ReLU → Linear(16→2) → [sigmoid, sigmoid]

Group C (PRC full): Same as B but β via tanh (β∈[-1,1])
  [α, β] = ... → [sigmoid, tanh]

Group D (Forced normal): Symmetrize FFN weight matrices
  For each Linear layer in FFN: W_effective = (W + W.T) / 2
  Apply symmetrization in forward pass (keep original W for gradient update,
  but use (W+W.T)/2 for actual computation)

TRAINING CONFIGURATION:
  - Optimizer: AdamW, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1
  - LR schedule: linear warmup 2000 steps, then cosine decay to 1e-5
  - Total steps: 50000
  - Batch size: start with 16, reduce if OOM. Use gradient accumulation 
    to get effective batch ~32 if needed.
  - Seeds: 3 per group (seeds 42, 123, 456)
  - Save checkpoint every 10000 steps
  - Evaluate on validation set every 2000 steps

CRITICAL: GEOMETRY LOGGING (runs every 1000 steps, inside torch.no_grad()):
  For each layer l (0 to 11), measure the FFN first linear layer weight W:
  
  1. henrici_l = henrici_number(W)  [formula as in Phase 1]
  2. curvature_l = ||[H, K]||_F     [formula as in Phase 1]
  3. excess_noise_l = ||W||²_F - Σ|λ_i|²
  4. frobenius_l = ||W||_F
  
  For PRC groups (B, C), also log per-layer:
  5. alpha_mean_l = mean α value over the last batch
  6. beta_mean_l = mean β value over the last batch
  
  Save to CSV: step, layer, henrici, curvature, excess_noise, frobenius, alpha, beta

TASK METRICS (every 2000 steps on validation set):
  1. Perplexity
  2. Loss (cross-entropy)
  3. Expected Calibration Error (ECE, 10 bins):
     - For each token prediction, bin by predicted probability
     - ECE = Σ (n_bin / n_total) * |accuracy_bin - confidence_bin|
  4. Confidence on incorrect predictions:
     - Collect max(softmax) for all WRONG next-token predictions
     - Report mean and fraction with confidence > 0.8
  
  Save to CSV: step, seed, group, perplexity, loss, ece, conf_on_errors, high_conf_error_frac

IMPORTANT IMPLEMENTATION NOTES:
  - Use torch.cuda.amp.autocast() for fp16 during forward/backward
  - Use torch.cuda.amp.GradScaler() for loss scaling
  - Geometry measurements should be done in fp32 for numerical stability:
    with torch.no_grad():
        W = layer.weight.float()  # cast to fp32 for eigenvalue computation
        # ... compute henrici, curvature, etc.
  - If eigenvalue computation is too slow on GPU, move W to CPU for eigvals:
    eigs = torch.linalg.eigvals(W.cpu())
  - For 512×512 matrices, eigenvalue decomposition takes ~10ms on CPU, 
    so doing it every 1000 steps for 12 layers = 12 × 10ms = negligible

ESTIMATED TIME: ~6-8 hours per seed, 12 seeds total = ~3-4 days
Run order: All 3 seeds of Group A first, then B, then C, then D.
This way if you need to stop early, you at least have complete results for A.
"""

# ============================================================
# PHASE 3: POST-TRAINING ANALYSIS
# ============================================================

PHASE_3 = """
After all Phase 2 training is complete, run post-training analysis on checkpoints.
This produces figures and tables for both papers.

ANALYSIS 3A: PRC PAPER RESULTS

Using the saved CSVs from Phase 2:

1. Main results table:
   For each group (A/B/C/D), report mean±std over 3 seeds:
   - Final perplexity
   - Final ECE
   - Mean confidence on errors
   
   Run paired t-tests: C vs A, C vs B, C vs D, B vs A
   Report p-values and Cohen's d.

2. β distribution figure:
   For Group C (PRC full), plot histogram of β values across all 12 layers
   at the final checkpoint. Mark negative values in red, positive in blue.
   
3. β evolution figure:
   For Group C, one seed: plot β per layer over training steps.
   12 lines (one per layer), x-axis = training step, y-axis = β value.
   Add a horizontal line at β=0.

4. Performance vs curvature scatter:
   For all groups/seeds, plot (mean curvature across layers, perplexity).
   One point per (group, seed). Color by group.
   Expected: U-shape or monotone, with Group C near optimal.

ANALYSIS 3B: NNOG PAPER RESULTS

1. Curvature evolution figure:
   From geometry CSVs: plot mean curvature across layers vs training step.
   One curve per group (A/B/C/D). Shows how training changes non-normality.

2. Per-layer curvature bar chart:
   At final checkpoint: bar chart of curvature per layer, grouped by model type.
   
3. Component non-normality:
   For Group A's final checkpoint, extract:
   - Attention weight matrices (W_Q, W_K, W_V, W_O): compute Henrici, curvature
   - FFN weight matrices (W_1, W_2): compute Henrici, curvature
   - Report in table: which component contributes most non-normality
   
4. Orthogonal init comparison:
   If time permits: train one more Group A model with orthogonal initialization
   (torch.nn.init.orthogonal_) and compare initial curvature + training curve.

ANALYSIS 3C: PSEUDOSPECTRUM VISUALIZATION

Take Group A's best seed, final checkpoint, layer 6 (middle layer) FFN first weight.
W is 2048×512 or 512×2048 depending on which FFN linear.
If W is not square, use W @ W.T (512×512) as the effective operator.

Compute pseudospectrum on a 150×150 grid:
  eigs = eigenvalues of W
  For grid points z in complex plane around eigenvalues:
    sigma_min = min singular value of (zI - W)
  Plot contours at σ_min = 0.05, 0.1, 0.2, 0.5
  Overlay eigenvalues as red dots.

This may take 30-60 minutes for a 512×512 matrix with 150×150 grid.
Save as high-res PDF.

ALL FIGURES: Save as PDF, 300 dpi, matplotlib with:
  plt.rcParams['font.size'] = 12
  plt.rcParams['font.family'] = 'serif'
  No titles (caption goes in LaTeX)
  Proper axis labels with units
  Legend inside figure
  Tight layout
"""

# ============================================================
# PHASE 4: HALLUCINATION / NOISE INJECTION
# ============================================================

PHASE_4 = """
Using Group A's best trained LM checkpoint, test the hallucination hypothesis.

SETUP:
  Load the trained Group A model (standard residual, best seed by perplexity).
  Use the validation set for evaluation.

NOISE INJECTION:
  We will inject noise matrices into the hidden state AFTER layer 6's FFN output,
  before the residual addition:
  
  Original: h = h + FFN(RMSNorm(h))
  Modified: h = h + FFN(RMSNorm(h)) + strength * h @ M.T
  
  where M is a d×d noise matrix (d=512), normalized to ||M||_op = 1.

  Three noise types:
  
  A) Growth noise: M = upper triangular random matrix
     M = torch.triu(torch.randn(512, 512), diagonal=1)
     M = M / torch.linalg.norm(M, ord=2)  # normalize operator norm to 1
     This has high transient growth potential.
  
  B) No-growth noise: M = strongly stable + off-diagonal
     D = torch.diag(-torch.rand(512) * 3 - 1)  # strongly negative diagonal
     U = torch.triu(torch.randn(512, 512), diagonal=1) * 0.3
     M = D + U
     M = M / torch.linalg.norm(M, ord=2)
     This has high excess noise but low transient growth.
  
  C) Normal noise: M = symmetric random matrix (control)
     A = torch.randn(512, 512)
     M = (A + A.T) / 2
     M = M / torch.linalg.norm(M, ord=2)

  Sweep strength in [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]

  For each (noise_type, strength), evaluate on 2000 validation tokens:
  
  Metrics:
  1. Perplexity
  2. Top-1 next-token accuracy
  3. Mean confidence on WRONG predictions: mean of max(softmax) where argmax ≠ target
  4. Fraction of wrong predictions with confidence > 0.8
  5. Error entropy: distribution of confusion patterns (are errors structured or random?)

OUTPUT:
  Save as CSV: noise_type, strength, perplexity, accuracy, conf_on_errors, high_conf_frac
  
  Generate figure:
  - 2 subplots side by side
  - Left: accuracy vs strength (3 curves: growth/no-growth/normal)
  - Right: confidence-on-errors vs strength (3 curves)
  - Expected: accuracy drops similarly for all types, but confidence-on-errors
    stays higher for growth noise than no-growth noise.

Verify noise matrix properties before starting:
  For each M, compute and print:
  - Henrici number
  - Curvature ||[H,K]||_F
  - Transient growth ratio (use scipy.linalg.expm to compute ||e^{tM}|| for t in [0,10])
"""

# ============================================================
# FILE ORGANIZATION
# ============================================================

FILE_STRUCTURE = """
Organize all outputs in this directory structure:

experiments/
├── phase1_rnn/
│   ├── rnn_training_log.csv        (epoch, loss, henrici, curvature, spec_radius)
│   ├── eigenvalues_epoch0.npy
│   ├── eigenvalues_final.npy
│   ├── fig_learning_nonnormality.pdf
│   ├── fig_pseudospectrum.pdf
│   └── fig_curvature_evolution.pdf
├── phase2_lm/
│   ├── geometry_log_groupA_seed42.csv
│   ├── geometry_log_groupA_seed123.csv
│   ├── ...  (one per group×seed)
│   ├── metrics_log_groupA_seed42.csv
│   ├── ...
│   └── checkpoints/
│       ├── groupA_seed42_step50000.pt
│       └── ...
├── phase3_analysis/
│   ├── main_results_table.csv
│   ├── statistical_tests.txt
│   ├── fig_beta_distribution.pdf
│   ├── fig_beta_evolution.pdf
│   ├── fig_performance_vs_curvature.pdf
│   ├── fig_curvature_per_layer.pdf
│   ├── fig_curvature_over_training.pdf
│   ├── fig_pseudospectrum_transformer.pdf
│   └── component_nonnormality_table.csv
└── phase4_hallucination/
    ├── noise_injection_results.csv
    ├── noise_matrix_properties.txt
    └── fig_hallucination_analysis.pdf
"""

# ============================================================
# MASTER PROMPT (give this to Claude Code first)
# ============================================================

MASTER_PROMPT = """
I'm running GPU experiments for two academic papers on non-normal operator 
geometry and residual connections in deep networks. Hardware: NVIDIA 4060Ti 
8GB VRAM, CUDA 12.6.

The experiments have 4 phases. Please run them in order. After each phase, 
save all results and report a summary before proceeding.

Key constraints:
- 8GB VRAM limit. Use fp16 mixed precision for Transformer training.
- Reproducibility: fixed seeds (42, 123, 456), save all random states.
- Statistical rigor: mean±std over seeds, paired t-tests, Cohen's d.
- All figures saved as PDF, publication quality (serif font, no titles, 
  proper labels, 300 dpi).
- Geometry measurements (eigenvalues, Henrici, curvature) must be computed 
  in fp32 even if model trains in fp16.
- Save all raw data as CSV for later analysis.

The theory predicts:
1. Non-normality (Henrici index, curvature) increases during learning
2. PRC with β∈[-1,1] may learn negative β in some layers (high noise layers)
3. Forced normalization (Group D) hurts perplexity significantly
4. Best model has intermediate curvature (not max, not zero)
5. Growth noise causes higher-confidence errors than no-growth noise
6. Softmax Jacobian is normal; attention/FFN Jacobians are non-normal

Please start with Phase 0 (setup + sanity check), then proceed to Phase 1.
"""

print("=" * 70)
print("  EXPERIMENT PLAN READY")
print("=" * 70)
print()
print("Files to give Claude Code:")
print("  1. This file (gpu_experiment_plan_v2.py) - full reference")
print("  2. Start with MASTER_PROMPT, then give each PHASE_N as needed")
print()
print("Estimated timeline:")
print("  Phase 0: 15 min (setup)")
print("  Phase 1: 30 min (RNN)")  
print("  Phase 2: 3-4 days (LM training, can run overnight)")
print("  Phase 3: 2-3 hours (analysis)")
print("  Phase 4: 1 hour (noise injection)")
print()
print("Phase 2 run order (to minimize risk of partial results):")
print("  Group A seed 42 → A seed 123 → A seed 456")
print("  → B seed 42 → B seed 123 → B seed 456")  
print("  → C seed 42 → C seed 123 → C seed 456")
print("  → D seed 42 → D seed 123 → D seed 456")
print()
print("After CIFAR-10 results come in, we can also add CIFAR")
print("geometry measurements to Phase 3 using existing checkpoints.")
