# RePersona

**Personality as an Emergent Property of Recursive Learning Under Structural Asymmetry**

## Overview

We propose that personality — the stable structure guiding an agent's decisions across situations — is not a pre-programmed attribute but an emergent property of recursive learning under structural asymmetry.

### Design Principles

1. **Values are strategies learned from experience** — not innate parameters
2. **Value differentiation requires structural asymmetry** — symmetric agents converge to identical policies
3. **Values follow a dynamic cycle with persistent scarring** — experiences leave lasting traces
4. **Recursive self-reinforcement** — each experience simultaneously contributes training data and reshapes the learning operator itself

### Formal Results

- **Memory Separation Theorem**: Episodic memory with retroactive credit assignment extends the effective learning horizon from zero to N steps
- **Three-Factor Decomposition Theorem**: Social differentiation requires episodic and emotional signals beyond core values

### Cognitive Architecture

A five-module architecture validated across multiple experimental settings:

| Module | Analogy | Role |
|--------|---------|------|
| **W** | Cortex | Core learning and representation |
| **H** | Hippocampus | Episodic memory |
| **F** | Amygdala | Emotional valuation |
| **P** | Prefrontal cortex | Working memory |
| **LLM** | Frozen language model | Language grounding |

### Experiments

- Multi-agent social games
- Spatial navigation: T-maze, radial arm maze, Morris water maze
- Dream-mediated memory consolidation
- Prediction-error-driven meta-learning
- Emotion vector decoding

All experiments use toy-scale simulations (525K parameters).

## Repository Structure

```
RePersona/
├── src/           # Core modules (W, H, F, P) and architecture
├── experiments/   # Experiment scripts and configurations
├── data/          # Datasets and experimental results
└── README.md
```

## License

TBD
