# ============================================================
# GROUP G: ADVERSARIAL NOISE DISCRIMINATOR
# Add to Phase 2 LM experiments after Groups A-D complete
# Hardware: 4060Ti 8GB VRAM
# ============================================================
#
# CONTEXT:
# Groups A-D are running (or completed). Group G uses Group A's
# trained checkpoint as a frozen base model. Only the discriminator
# and gate are trained.
#
# CPU-SCALE RESULTS (toy classification, 8 seeds):
#   Group G vs A: ΔAccuracy = -0.0002 (unchanged)
#                 ΔConfOnErr = -0.025 (5.2% reduction)
#                 p < 0.0001, Cohen's d = -21
#   Conclusion: G selectively suppresses amplified noise without
#   hurting accuracy. Now verify on Transformer LM.
#
# ESTIMATED TIME: ~6-8 hours (3 seeds × ~2 hours each)

GROUP_G_PROMPT = """
After Groups A-D training is complete, run Group G experiments.

GROUP G: Adversarial Noise Discriminator

CONCEPT:
  Train a small auxiliary network (discriminator) to predict which 
  hidden-state dimensions are associated with incorrect predictions.
  Use the discriminator's output to selectively suppress those 
  dimensions, reducing high-confidence errors without hurting 
  overall perplexity.

  PRC suppresses noise UNIFORMLY (it doesn't know what's noise).
  The discriminator learns what amplified noise LOOKS LIKE by 
  observing which hidden-state patterns lead to errors.

SETUP:
  1. Load Group A's best checkpoint (frozen, all weights fixed)
  2. Add a discriminator module at layer 3 (middle of 6 layers)
  3. Train ONLY the discriminator + gate parameters

DISCRIMINATOR ARCHITECTURE:
  class NoiseDiscriminator(nn.Module):
      def __init__(self, d_model=512, d_hidden=64):
          super().__init__()
          self.net = nn.Sequential(
              nn.Linear(d_model, d_hidden),
              nn.ReLU(),
              nn.Linear(d_hidden, d_hidden),
              nn.ReLU(),
              nn.Linear(d_hidden, d_model),
              nn.Sigmoid()  # output: per-dimension noise score in [0,1]
          )
          self.gate_scale = nn.Parameter(torch.tensor(0.0))
          # gate_scale starts at 0 (no suppression) and learned
      
      def forward(self, h):
          noise_score = self.net(h.detach())  # detach: don't backprop into main model
          gate = 1.0 - torch.sigmoid(self.gate_scale) * noise_score
          return h * gate, noise_score

INTEGRATION INTO TRANSFORMER:
  In the forward pass of the frozen Group A model, at layer 3's 
  FFN output (BEFORE the residual addition):
  
  # Original: h = h + FFN(RMSNorm(h))
  # Modified: h = h + gate(FFN(RMSNorm(h)))
  
  def forward_with_discriminator(self, x):
      for i, layer in enumerate(self.layers):
          h_norm = layer.norm(x)
          h_attn = layer.attention(h_norm)
          x = x + h_attn
          
          h_norm2 = layer.norm2(x)
          h_ffn = layer.ffn(h_norm2)
          
          if i == 3:  # apply discriminator at layer 3
              h_ffn, noise_score = self.discriminator(h_ffn)
          
          x = x + h_ffn
      return self.head(x)

TRAINING PROCEDURE (two-phase alternating):

  Phase 1 (steps 1-1000): Warmup - train discriminator only
    - Run frozen model forward, collect hidden states at layer 3
    - For each token, check if model's prediction is correct
    - Train discriminator to predict correctness:
      
      loss_disc = 0
      for each token t in batch:
          noise_score = discriminator.net(h_layer3[t].detach())
          if prediction[t] is CORRECT:
              # Noise score should be low (this is signal, not noise)
              loss_disc += torch.mean(noise_score ** 2)
          else:
              # Noise score should be high (something here caused error)
              loss_disc += torch.mean((1 - noise_score) ** 2)
      
      # Add sparsity regularization (don't flag everything as noise)
      loss_disc += 0.01 * torch.mean(noise_score)
      
      optimizer_disc.zero_grad()
      loss_disc.backward()
      optimizer_disc.step()
    
    gate_scale stays at 0 during this phase (no suppression yet).

  Phase 2 (steps 1000-5000): Joint training with gating
    - Gradually increase gate_scale (learned, but starts small)
    - Alternate every 50 steps:
      
      40 steps: Train main loss (cross-entropy) with gating active
        - Forward through frozen model + discriminator gate
        - Backprop only through gate_scale (not through discriminator.net)
        - This teaches gate_scale how much suppression is optimal
      
      10 steps: Update discriminator
        - Same as Phase 1 but with gating active
        - Discriminator adapts to the gated hidden states

TRAINABLE PARAMETERS:
  discriminator.net: 512*64 + 64*64 + 64*512 = 69,632
  gate_scale: 1
  Total: ~70K (vs ~12M frozen base model = 0.6% extra)

OPTIMIZER:
  Adam, lr=1e-4 for discriminator, lr=1e-3 for gate_scale
  No weight decay on discriminator (it's tiny)

TRAINING DATA:
  Same WikiText-103 train set, same batch size, same context length

SEEDS: 42, 123, 456 (same as Groups A-D)

METRICS (evaluate every 500 steps on validation set):
  1. Perplexity (should stay close to Group A's ~51)
  2. ECE
  3. Confidence on errors (PRIMARY METRIC - should decrease)
  4. High-confidence error fraction (conf > 0.8 on wrong tokens)
  5. Discriminator accuracy (how well does it predict errors?)
  6. Mean noise_score on correct vs incorrect predictions
  7. gate_scale value (how much suppression did model learn?)
  8. Per-dimension noise_score histogram (which dims get flagged?)

Save to CSV:
  metrics_log_groupG_seed{N}.csv: step, perplexity, ece, 
    conf_on_errors, high_conf_error_frac, disc_accuracy,
    mean_noise_score_correct, mean_noise_score_incorrect, gate_scale

EXPECTED RESULTS (based on CPU toy experiments):
  - PPL ≈ Group A (51.2 ± 0.3) — no degradation
  - ConfOnErr significantly lower than Group A (0.226)
  - ConfOnErr lower than Group B (0.231) — B increases confidence!
  - High-conf error fraction reduced
  - Discriminator learns to assign higher noise_score to incorrect 
    predictions than correct ones

COMPARISON TABLE (to generate after all groups complete):
  | Group | PPL | ConfOnErr | HighConfErr | Method |
  |-------|-----|-----------|-------------|--------|
  | A     | 51.2| 0.226     | 0.012       | Standard |
  | B     | 49.8| 0.231     | ?           | PRC β≥0 (better PPL, worse conf) |
  | C     | ?   | ?         | ?           | PRC full (pending) |
  | D     | ?   | ?         | ?           | Forced normal (pending) |
  | G     | ~51 | <0.226    | <0.012      | Discriminator (targeted) |

KEY COMPARISON: B vs G
  - B improves PPL but INCREASES ConfOnErr (more confident errors)
  - G preserves PPL and DECREASES ConfOnErr (fewer confident errors)
  - This directly demonstrates: uniform amplification (B) vs 
    targeted suppression (G) have opposite effects on reliability

IMPLEMENTATION NOTES:
  - discriminator.net input is DETACHED from main model graph
    (no gradient flows from discriminator loss into base model)
  - gate output multiplies h_ffn BEFORE residual addition
  - Use fp16 for base model forward, fp32 for discriminator
  - If VRAM is tight, only apply discriminator to layer 3 
    (not all layers). Can extend to all layers later if it works.

ABLATIONS (if time permits, run 1 seed each):
  G1: Discriminator at layer 1 (early) instead of layer 3
  G2: Discriminator at layer 5 (late) instead of layer 3
  G3: Discriminator at ALL layers (one shared discriminator)
  G4: Discriminator without sparsity regularization
  G5: gate_scale fixed at 0.3 (not learned)
  
  These tell us:
  - Which layer benefits most from targeted suppression?
  - Does a shared discriminator generalize across layers?
  - How important is the sparsity prior?
"""

# ============================================================
# TIMELINE
# ============================================================

TIMELINE = """
WHEN TO RUN:
  After Groups A-D are all complete.
  Group G uses Group A checkpoint, so A must finish first.
  
  If Groups C and D are still running, you can start Group G 
  in parallel using a separate GPU process (if VRAM allows) 
  or wait for C/D to finish.

SEQUENCE:
  1. Load Group A best checkpoint
  2. Run Group G seed 42 (~2 hours)
  3. Quick check: is ConfOnErr decreasing? If yes, continue.
  4. Run Group G seeds 123 and 456
  5. Statistical comparison with all groups

TOTAL TIME: ~6-8 hours
"""

print("Group G experiment plan ready.")
print()
print("Key points:")
print("  - Uses Group A checkpoint (frozen)")
print("  - Only trains ~70K discriminator params")  
print("  - Target: reduce ConfOnErr without hurting PPL")
print("  - CPU test showed: ΔConfOnErr = -5.2%, p < 0.0001")
print("  - Estimated GPU time: ~6-8 hours (3 seeds)")
print()
print("Give GROUP_G_PROMPT to Claude Code after Groups A-D complete.")
