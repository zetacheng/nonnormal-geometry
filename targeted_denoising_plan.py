# ============================================================
# TARGETED NOISE SUPPRESSION EXPERIMENTS
# Groups E, F, G: Beyond uniform PRC
# Hardware: 4060Ti 8GB VRAM
# ============================================================
#
# CONTEXT: PRC (Groups B, C) suppresses noise uniformly via 
# low-rank projection. This helps with background noise but 
# cannot distinguish signal amplified by transient growth from 
# noise amplified by transient growth. Both look the same in 
# hidden-state space after amplification.
#
# GOAL: Test three methods for SELECTIVE noise suppression that
# can identify and discard amplified noise while preserving 
# amplified signal.
#
# BASELINE COMPARISONS (already trained):
#   Group A: Standard residual (PPL ~51.2)
#   Group B: PRC β≥0 (PPL ~49.8) 
#   Group C: PRC full β∈[-1,1] (running)
#   Group D: Forced normal (running)
#
# NEW GROUPS:
#   Group E: Perturbation consistency check
#   Group F: Schur direction projection  
#   Group G: Adversarial noise discriminator

# ============================================================
# GROUP E: PERTURBATION CONSISTENCY CHECK
# ============================================================

GROUP_E = """
GROUP E: Perturbation Consistency Gating

THEORY:
  Signal comes from data structure → stable under small perturbation.
  Amplified noise comes from transient growth of random components
  → unstable under small perturbation.
  
  If we forward-pass twice with slightly different dropout masks 
  (or small Gaussian perturbation), the DIFFERENCE between the two 
  hidden states reveals which components are noise-driven.

ARCHITECTURE:
  Same 6-layer Transformer as Group A. On EACH layer's FFN output,
  add a consistency gate:

  def forward_with_consistency(self, x):
      # Normal FFN computation
      h1 = self.ffn(self.norm(x))
      
      # Perturbed FFN computation (shared weights, different noise)
      with torch.no_grad():
          noise = torch.randn_like(x) * 0.01 * x.norm(dim=-1, keepdim=True)
          h2 = self.ffn(self.norm(x + noise))
      
      # Consistency score per hidden dimension
      # High consistency = signal, low consistency = amplified noise
      consistency = 1.0 - torch.abs(h1 - h2) / (torch.abs(h1) + 1e-8)
      consistency = consistency.detach()  # don't backprop through this
      
      # Gate: suppress inconsistent (noise) components
      gate = torch.sigmoid(self.gate_bias + self.gate_scale * consistency)
      # gate_bias and gate_scale are learnable per-layer scalars
      
      h_gated = h1 * gate
      
      return x + h_gated

TRAINABLE PARAMETERS: 
  2 scalars per layer (gate_bias, gate_scale) = 12 total.
  All other weights identical to Group A.
  
  Alternatively, make gate a small MLP:
  gate = MLP(consistency)  where MLP: d→16→d with sigmoid output
  This adds ~16K params per layer = ~100K total (still tiny).

TRAINING:
  Same as Group A. The extra forward pass for h2 is in no_grad,
  so backward pass cost is unchanged. Forward cost is ~2x per layer
  for FFN (one normal + one perturbed), so total training is ~1.5x 
  slower. 
  
  VRAM: need to store h1 and h2 simultaneously. For d=512 this is 
  negligible.

CRITICAL DETAIL:
  The perturbation magnitude 0.01 * ||x|| is a hyperparameter.
  Too small: can't distinguish signal from noise.
  Too large: everything looks inconsistent.
  Start with 0.01, sweep [0.001, 0.005, 0.01, 0.02, 0.05] if needed.

EXPECTED RESULTS:
  - PPL close to Group B (preserves signal, only suppresses noise)
  - ConfOnErr LOWER than Group B (suppresses amplified noise that 
    causes high-confidence errors)
  - Consistency scores should be lower in deeper layers (more 
    noise accumulation) and lower for incorrectly predicted tokens

METRICS (in addition to standard PPL/ECE/ConfOnErr):
  - Per-layer mean consistency score
  - Consistency score on correct vs incorrect predictions
  - Gate activation patterns (which dimensions get suppressed)
"""

# ============================================================
# GROUP F: SCHUR DIRECTION PROJECTION
# ============================================================

GROUP_F = """
GROUP F: Schur Direction Projection (Geometric Denoising)

THEORY:
  The Schur decomposition T = U(Λ + M)U* separates the operator 
  into its normal part (eigenvalues Λ) and non-normal part 
  (strictly upper triangular M). The columns of U corresponding 
  to large entries in M are the "non-normal directions" — these 
  are the directions along which transient growth amplifies noise.
  
  If we project AWAY from the top-k non-normal directions, we 
  remove the most noise-prone components while keeping the 
  eigenvalue-aligned (normal) signal.

ARCHITECTURE:
  Same 6-layer Transformer as Group A. After training, compute 
  the Schur decomposition of each FFN weight matrix W1 (512×2048 
  reshaped or W1^T @ W1 for 512×512).
  
  Actually, computing Schur on the fly during training is too 
  expensive. Instead, use a LEARNED approximation:

  def forward_with_schur_proj(self, x):
      h = self.ffn(self.norm(x))
      
      # Learned "non-normal direction" projector
      # U_noise: d × k matrix (k=16 or 32 directions to suppress)
      # Projects h onto non-normal subspace, then subtracts
      proj = h @ self.U_noise @ self.U_noise.T  # project onto noise subspace
      h_clean = h - self.suppress_scale * proj
      
      return x + h_clean

  where U_noise ∈ R^{d×k} is learned (initialized from actual 
  Schur vectors of the pretrained Group A weights), and 
  suppress_scale is a learnable scalar per layer.

INITIALIZATION:
  Before training Group F, take Group A's trained checkpoint.
  For each layer's FFN W1:
    1. Compute W = W1^T @ W1 (512×512)
    2. Schur decomposition: W = U(Λ+M)U*
    3. Find the k columns of U with largest ||M_col|| 
       (most non-normal directions)
    4. Initialize U_noise with these k columns
  
  Then train with U_noise learnable (can drift from initialization).

TRAINABLE PARAMETERS:
  U_noise: d × k per layer. If k=16: 512×16 = 8192 per layer = ~50K total.
  suppress_scale: 1 per layer = 6 total.
  All other weights same as Group A (can be frozen or fine-tuned).

  Option 1: Freeze Group A weights, only train U_noise + suppress_scale.
  Option 2: Fine-tune everything (more expensive but potentially better).
  Start with Option 1 for clean comparison.

TRAINING:
  If freezing base weights: only 50K trainable params, very fast.
  If fine-tuning: same cost as Group A.
  No extra forward pass needed (unlike Group E).

EXPECTED RESULTS:
  - Should suppress amplified noise more selectively than PRC
  - PPL may be slightly worse than Group B (we're removing some 
    capacity) but ConfOnErr should be much lower
  - The learned U_noise should correlate with the Schur vectors 
    but may discover better noise directions through training

METRICS:
  - Angle between learned U_noise and initial Schur vectors 
    (do they stay aligned or drift?)
  - Per-layer suppress_scale (which layers need most suppression?)
  - Overlap between suppressed subspace and the subspace of 
    hidden-state perturbations that cause prediction changes
"""

# ============================================================
# GROUP G: ADVERSARIAL NOISE DISCRIMINATOR
# ============================================================

GROUP_G = """
GROUP G: Adversarial Noise Discriminator

THEORY:
  Train a small auxiliary network (discriminator) to predict 
  whether a given hidden state will lead to a correct or incorrect 
  prediction. Use the discriminator's output to selectively 
  suppress "error-prone" hidden-state components.
  
  This is the most aggressive approach: it directly learns what 
  "amplified noise" looks like in hidden-state space, rather than 
  relying on geometric proxies.

ARCHITECTURE:
  Same 6-layer Transformer as Group A (main model).
  Add a discriminator at layer 3 (middle layer, highest information):

  class NoiseDiscriminator(nn.Module):
      def __init__(self, d=512, d_hidden=64):
          self.net = nn.Sequential(
              nn.Linear(d, d_hidden),
              nn.ReLU(),
              nn.Linear(d_hidden, d_hidden),
              nn.ReLU(),
              nn.Linear(d_hidden, d)  # per-dimension "noise score"
          )
          self.gate_scale = nn.Parameter(torch.tensor(0.0))
      
      def forward(self, h, suppress=True):
          noise_score = torch.sigmoid(self.net(h.detach()))
          # noise_score[i] ≈ 1 means dimension i is likely noise
          # noise_score[i] ≈ 0 means dimension i is likely signal
          
          if suppress:
              gate = 1.0 - self.gate_scale * noise_score
              return h * gate
          return noise_score

TRAINING PROCEDURE (two-phase per step):
  
  Phase 1: Train discriminator (freeze main model)
    - Forward pass through main model, get hidden state h at layer 3
    - Get model prediction (argmax of output logits)
    - Label: correct=0, incorrect=1
    - For CORRECT predictions: noise_score should be low (all signal)
    - For INCORRECT predictions: noise_score should be high 
      (some dimensions are noise-driven)
    
    Discriminator loss:
      For correct tokens: L_correct = ||noise_score||^2  (push to 0)
      For incorrect tokens: L_incorrect = -log(max(noise_score))  
        (at least one dimension should be flagged as noise)
      
      Actually, simpler formulation:
      L_disc = BCE(max(noise_score), is_incorrect)
      i.e., discriminator predicts whether the token will be wrong
      based on the noise pattern in the hidden state.
      
      BUT we want per-dimension scores, not just a binary prediction.
      Better: use a reconstruction-based approach.
      
      For correct tokens: h should be fully reconstructable from 
        the "clean" subspace → minimize ||h - h*(1-noise_score)||^2
      For incorrect tokens: h has noise components that should be 
        identified → maximize the discriminator's ability to 
        predict the error direction

    REVISED (simpler) discriminator loss:
      Run main model forward → get predictions
      Mark each token as correct (c=1) or incorrect (c=0)
      
      L_disc = Σ_correct ||noise_score(h)||^2     (don't flag signal)
             + Σ_incorrect -||noise_score(h)||^2   (do flag noise)
             + λ * ||noise_score||^1               (sparsity: flag few dims)

  Phase 2: Train main model (freeze discriminator)
    - Normal forward pass with discriminator gating active
    - h_gated = h * (1 - gate_scale * noise_score.detach())
    - Standard cross-entropy loss on output
    - gate_scale starts at 0 and slowly increases (warmup)

TRAINING SCHEDULE:
  Steps 1-2000: Train main model only (warmup, no discriminator)
  Steps 2000-5000: Train discriminator only (freeze main model)
  Steps 5000-20000: Alternate every 100 steps:
    - 80 steps: train main model with gating
    - 20 steps: update discriminator

TRAINABLE PARAMETERS:
  Discriminator: 512*64 + 64*64 + 64*512 ≈ 70K params
  gate_scale: 1 scalar
  Main model: same as Group A (~12M)

TRAINING COST:
  ~1.3x Group A (discriminator forward/backward is small,
  but the alternating schedule adds overhead)

EXPECTED RESULTS:
  - Best ConfOnErr among all groups (directly optimized to find errors)
  - PPL should be close to Group A or slightly better (we're only 
    suppressing noise, not useful signal)
  - If it works, the discriminator's learned noise_score pattern 
    reveals WHAT amplified noise looks like — this is scientifically 
    valuable regardless of PPL improvement

ANALYSIS:
  - Visualize noise_score patterns: are certain dimensions 
    consistently flagged as noise?
  - Compare flagged dimensions to Schur non-normal directions 
    (overlap with Group F's approach?)
  - Does noise_score correlate with per-dimension variance of 
    hidden states across similar inputs? (high variance = noise)
  - Do the flagged dimensions correspond to high-curvature 
    directions in [H,K]?
"""

# ============================================================
# COMPARISON PROTOCOL
# ============================================================

PROTOCOL = """
EXPERIMENTAL PROTOCOL FOR GROUPS E, F, G

All groups use the SAME base architecture as Group A (6-layer 
decoder-only Transformer, d=512, WikiText-103).

Two options for training:
  OPTION 1 (RECOMMENDED): Start from Group A's trained checkpoint.
    Freeze base weights. Only train the new components (gate, 
    U_noise, discriminator). 5000-10000 steps of fine-tuning.
    This isolates the effect of the noise suppression mechanism.
    
  OPTION 2: Train from scratch like all other groups (20000 steps).
    More expensive but cleaner comparison.

Start with Option 1 for all three. If promising, run Option 2 
for the best-performing group.

METRICS (all groups):
  1. Perplexity (PPL)
  2. ECE (Expected Calibration Error)
  3. Confidence on errors (mean max-softmax for wrong predictions)
  4. High-confidence error fraction (conf > 0.8 on wrong predictions)
  5. Per-layer curvature and excess noise

ADDITIONAL METRICS:
  Group E: consistency scores (correct vs incorrect tokens)
  Group F: angle(learned U_noise, Schur vectors), suppress_scale
  Group G: noise_score patterns, discriminator accuracy

STATISTICAL TESTS:
  All pairwise comparisons with Groups A, B, C using paired t-test 
  + Cohen's d (3 seeds each).

PRIORITY ORDER:
  1. Group E (simplest to implement, strongest theoretical backing)
  2. Group G (most powerful if it works, but most complex)
  3. Group F (requires Schur decomposition, may be fragile)

ESTIMATED TIME:
  Option 1 (fine-tune from checkpoint):
    Group E: ~3 hours (2x forward cost for consistency check)
    Group F: ~2 hours (no extra forward cost)
    Group G: ~4 hours (alternating training schedule)
    Total: ~9 hours for all three, single seed
    With 3 seeds: ~27 hours

  Option 2 (train from scratch):
    Each group: ~6-8 hours × 3 seeds = ~18-24 hours per group
    Total: ~54-72 hours
"""

# ============================================================
# MASTER PROMPT FOR CLAUDE CODE
# ============================================================

MASTER_PROMPT = """
I'm running targeted noise suppression experiments for an academic paper.
These extend the PRC experiments already completed (Groups A-D).

I have a trained Group A checkpoint (6-layer Transformer, d=512, 
WikiText-103, PPL~51.2). I want to test three new noise suppression 
methods by fine-tuning from this checkpoint.

Hardware: 4060Ti 8GB VRAM, CUDA 12.6.
All base model weights are FROZEN. Only the new components are trained.
Training: 5000 steps, same data/batch/optimizer as original training.
Seeds: 42, 123, 456.

Please implement Group E first (Perturbation Consistency Gating), 
then Group F (Schur Direction Projection), then Group G (Adversarial 
Noise Discriminator).

For each group, save:
  - metrics_log (PPL, ECE, conf_on_errors at each eval step)
  - Any group-specific metrics (consistency scores, noise_scores, etc.)

Key constraint: these experiments should be COMPARABLE to Groups A-D,
so use the exact same evaluation code and validation set.
"""

print("=" * 60)
print("  TARGETED NOISE SUPPRESSION: Groups E, F, G")
print("=" * 60)
print()
print("Group E: Perturbation Consistency Gating")
print("  - 2 forward passes, gate inconsistent dimensions")
print("  - Theory: signal is stable, amplified noise is not")
print("  - Extra params: ~100K, Extra cost: ~1.5x forward")
print()
print("Group F: Schur Direction Projection")
print("  - Project away from non-normal Schur directions")
print("  - Theory: noise amplification along specific directions")
print("  - Extra params: ~50K, Extra cost: ~1.0x (minimal)")
print()
print("Group G: Adversarial Noise Discriminator")  
print("  - Learn to predict which dimensions cause errors")
print("  - Theory: directly identify amplified noise patterns")
print("  - Extra params: ~70K, Extra cost: ~1.3x forward")
print()
print("All fine-tuned from Group A checkpoint (frozen base weights)")
print("Priority: E > G > F")
print("Total time: ~27 hours (3 groups × 3 seeds)")
