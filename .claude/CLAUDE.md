# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Architecture

Three-component JEPA system. See [`docs/architecture.html`](docs/architecture.html) for the data flow diagram.

| Component | Frozen? | Notes |
|---|---|---|
| Price Encoder (context) | Yes (initially) | Moirai preferred; unfreeze top 2 layers only if domain shift confirmed |
| Target Encoder | Always | EMA copy of price encoder — **never receives gradients** |
| Text Encoder | Always | FinBERT / frozen Llama-3.2-1B |
| Predictor | No — trained | Option A: lightweight transformer. Option B: Qwen2.5-1.5B + LoRA (preferred) |

**Predictor wiring (Option B):** `z_price` and `z_text` are projected into the LM's embedding dim and prepended as prefix tokens. A learned query token is appended; its final hidden state is `z_pred`.

---

## Critical Implementation Rules

These are non-obvious constraints that cause silent failures if violated.

**1. LayerNorm after every projection layer.**
Financial latents and LM embedding spaces have incompatible scale and geometry. Missing LayerNorm causes unstable early training. No exceptions.

**2. Drop the VICReg invariance term.**
Standard VICReg has three terms: variance (V), invariance (I), covariance (C). Use only V + C. The invariance term conflicts with the JEPA temporal loss because context and target windows are different time slices, not augmentations.

**3. EMA target encoder receives zero gradients.**
`target_params = ema_decay * target_params + (1 - ema_decay) * context_params` — update happens after each step, outside the optimizer. If gradients accidentally flow through it, the self-supervised signal collapses.

**4. VICReg requires batch size ≥ 256.**
Covariance estimates are noisy below this. Use gradient accumulation if GPU memory is the constraint.

**5. Initialize codebook with k-means, not randomly.**
Run a frozen forward pass, collect `z_price` embeddings, run k-means, use centroids as initial codebook vectors. Random init causes codebook collapse (many dead vectors).

**6. Warm up projection layers before enabling LoRA (Option B predictor).**
Steps 1–500: train projections only, freeze LoRA. Steps 500+: enable LoRA at `3e-5` lr, projections at `1e-4`.

---

## Loss

```
L_total = L_jepa + λ_v * V(z_price) + λ_c * C(z_price)
```
- `L_jepa = 1 - cosine_similarity(z_pred, z_target)` (or L2/d)
- `λ_v = 25.0`, `λ_c = 1.0` (starting values)
- Apply regularization to `z_price` only, not `z_pred` or `z_text`

If using codebook: replace `z_price` with `z_codebook` and add:
```
L_total += λ_commit * ||z_price - sg(z_codebook)||²
```

---

## Training Phases

| Phase | Steps | What's active |
|---|---|---|
| 1 | 1–500 | Projection layers only. Everything else frozen. |
| 2 | 500+ | Projection layers + LoRA adapters. JEPA loss + regularization. |
| 3 | Optional | Unfreeze top 2 price encoder layers at 0.1× lr if val loss plateaus. |

---

## Available ECC Tools (globally installed in ~/.claude/)

ECC (Everything Claude Code) is installed globally. Use these for this project:

**Agents** (invoke via `Agent` tool with `subagent_type`):
- `mle-reviewer` — ML code review: architecture, training loops, loss functions
- `code-reviewer` — General code review
- `python-reviewer` — Python/PEP 8/type hints
- `security-reviewer` — Security audit
- `architect` — System design decisions
- `planner` — Task breakdown before implementation

**Skills most relevant here** (invoke via `/skill-name`):
- `/python-review` — Python code review
- `/tdd-workflow` — Test-driven development loop
- `/eval-harness` — Model evaluation pipeline
- `/security-review` — Security checklist
- `/mle-workflow` — Full ML experiment workflow
- `/continuous-learning-v2` — Pattern extraction after sessions
- `/plan` — Feature planning with risk assessment
- `/code-review` — Review uncommitted changes

**Active hooks** (run automatically, no action needed):
- `SessionStart` — loads previous context
- `PreToolUse` — secret detection, config protection, fact-forcing gate (GateGuard blocks first edit per file; present facts then retry)
- `PostToolUse` — quality gate after file edits, context monitoring
- `Stop` — session state persistence, pattern evaluation, cost tracking

ECC scripts live at `~/.claude/scripts/` and are resolved automatically by hooks.

---

## File Structure

```
jepa_quant/
  encoders/
    price_encoder.py       # Moirai/TimesFM wrapper + projection head
    text_encoder.py        # FinBERT/Llama wrapper + projection head
    target_encoder.py      # EMA wrapper — no grad, post-step update only
  predictor/
    transformer_predictor.py   # Option A
    lm_predictor.py            # Option B (preferred)
  regularization/
    vicreg.py              # V + C terms only
    codebook.py            # Soft codebook + commitment loss
  training/
    jepa_loss.py
    train.py               # Phased training loop
    ema.py
  data/
    price_dataset.py
    text_dataset.py
    conditioning_dataset.py
```

---

## Training Diagnostics to Log

- Codebook utilization (fraction of active vectors per batch) — if <50%, reduce `λ_commit`
- Per-dimension std of `z_price` — variance collapse early warning
- JEPA loss vs regularization loss breakdown

---

## Self-Improvement Protocol

After completing any non-trivial task, update this file if any of the following apply — **only if every future agent in this codebase would need to know it**:

1. **Gotchas encountered**: silent failures, wrong assumptions about the architecture, hyperparameter interactions that weren't obvious
2. **Loops that required user intervention**: if you got stuck and the user had to redirect you, record what the trap was and how to avoid it
3. **Filter before writing**: ask "will every agent working in this codebase need to know this?" — if no, do not add it
4. **Update Project Status below** to reflect the current phase, what's been decided, and what's still open

Use HTML files in `docs/` for any concept that benefits from a diagram. Reference them from this file.

---

## Project Status

> **Maintained by the self-improvement protocol. Update this after each task.**

- **Phase**: Exploration
- **Open decision**: Anti-collapse regularization strategy — VICReg (V+C only) vs soft codebook bottleneck. Start with VICReg; add codebook if PCA/UMAP shows poor regime separation.
- **Decided**: Drop VICReg invariance term (conflicts with JEPA temporal objective). Prefer Qwen2.5-1.5B + LoRA as predictor (Option B).

### Log

**2026-06-02 — ECC setup**
- Gotcha: GateGuard has three distinct triggers per session, each requiring facts before retrying:
  1. **First Bash** — state the user request and what the command produces
  2. **First Edit/Write per file** (including new files) — state what imports it, no duplicate exists, data fields, user instruction
  3. **Destructive Bash** (git rm, reset, etc.) — state files affected, one-line rollback, user instruction
  All gates pass on the second attempt after facts are presented.
