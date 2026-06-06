# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Cost is not a concern.** The user is on a monthly (flat-rate) plan, so session cost warnings are irrelevant here -- do not factor spend into decisions, suggest cheaper shortcuts to save money, or wrap up early to limit cost. Optimize purely for correctness and getting the task done well.

> **Colab compute budget: 250 compute units/month.** Treat GPU-hours as a limited resource -- prefer efficient training runs, avoid redundant full re-trains, and don't burn units on debugging that can be done CPU-side first.

---

## Research Workflow (Mandatory for All Notebooks and Experiments)

Every notebook and experiment script MUST open with a structured header cell using this exact template, filled in BEFORE any code runs:

```
QUESTION: the single thing this experiment decides
H1 / H0: the hypothesis and its null, stated so they are distinguishable by the result
MEASURED TARGET: exact metric, exact dataset/split, exact computation (no vague "val_jepa" --
  say which batches, how many, what seed)
DECISION RULE: numeric thresholds. e.g. "if shuffled cosine >= 0.8 * true cosine, conclude degenerate"
PRIORS / ASSUMPTIONS: what we believe going in and why, what the inputs are assumed to be
  (raw vs returns), known confounds
FALSIFIER: the observation that would kill H1
RESULT: (filled after running)
DECISION + NEXT ACTION: (filled after running)
```

Inline, every nontrivial implementation choice gets a one-line reason in a comment or markdown cell:
why this normalization, why this baseline, why this threshold. No silent decisions.

Reusable logic goes in modules under `src/jepa_quant/eval/`, not buried in notebook cells.
Notebooks are thin drivers: import, call, print numbers, fill RESULT and DECISION.

---

## Metric Discipline

**The primary metric is the downstream linear probe, not val_jepa.**

val_jepa = 1 - cosine_similarity(z_pred, z_target) is an intrinsic metric. It can be low
(high cosine similarity) via EMA collapse: if the projection heads of the context encoder
and EMA target encoder converge to similar mappings regardless of input window, then every
z_pred and z_target pair will be cosine-similar, and val_jepa will be low even when the
representation carries no temporal structure.

**Before reporting any val_jepa improvement as progress, you MUST:**
1. Run `shuffled_target_control` (in `src/jepa_quant/eval/diagnostics.py`). Compute
   ratio = shuffled_cosine_mean / true_cosine_mean. If ratio >= 0.8, the improvement is
   degenerate and should not be reported as progress.
2. Compare against the untrained baseline (fresh random init, same arch). If trained
   val_jepa >= untrained val_jepa, training did not help.

**No new training run is justified until a shuffled-target control and the relevant baselines
exist for whatever is being claimed.** A new architecture change or hyperparameter sweep that
cannot beat the shuffled control is not a forward step.

---

## Architecture

Three-component JEPA system. See [`docs/architecture.html`](docs/architecture.html) for the data flow diagram.

| Component | Frozen? | Notes |
|---|---|---|
| Price Encoder (context) | Yes (initially) | **Default backend = `transformer`** (lightweight, native numpy 2). Moirai is architecturally preferred but its `uni2ts` dep forces numpy<2 and is unusable on stock Colab. Only run `backend='moirai'` in a clean numpy<2 env. Unfreeze top 2 layers only if domain shift confirmed. |
| Target Encoder | Always | EMA copy of price encoder -- **never receives gradients** |
| Text Encoder | Always | FinBERT / frozen Llama-3.2-1B |
| Predictor | No -- trained | Option A: lightweight transformer. Option B: Qwen2.5-1.5B + LoRA (preferred) |

**Predictor wiring (Option B):** `z_price` and `z_text` are projected into the LM's embedding dim and prepended as prefix tokens. A learned query token is appended; its final hidden state is `z_pred`.

**Input pipeline:** Raw OHLCV parquets -> log-returns (`_timestep_features` in price_dataset.py) -> per-sample normalize by context window mean/std (causal: stats from context only, applied to both context and target). Inputs to the encoder are normalized log-returns, NOT raw prices. Price level does NOT enter the encoder.

---

## Critical Implementation Rules

These are non-obvious constraints that cause silent failures if violated.

**1. LayerNorm after every projection layer.**
Financial latents and LM embedding spaces have incompatible scale and geometry. Missing LayerNorm causes unstable early training. No exceptions.

**2. Drop the VICReg invariance term.**
Standard VICReg has three terms: variance (V), invariance (I), covariance (C). Use only V + C. The invariance term conflicts with the JEPA temporal loss because context and target windows are different time slices, not augmentations.

**3. EMA target encoder receives zero gradients.**
`target_params = ema_decay * target_params + (1 - ema_decay) * context_params` -- update happens after each step, outside the optimizer. If gradients accidentally flow through it, the self-supervised signal collapses.

**4. VICReg requires batch size >= 256.**
Covariance estimates are noisy below this. Use gradient accumulation if GPU memory is the constraint.

**5. Initialize codebook with k-means, not randomly.**
Run a frozen forward pass, collect `z_price` embeddings, run k-means, use centroids as initial codebook vectors. Random init causes codebook collapse (many dead vectors).

**6. Warm up projection layers before enabling LoRA (Option B predictor).**
Steps 1-500: train projections only, freeze LoRA. Steps 500+: enable LoRA at `lr_adapter` lr, projections at `lr_proj`.

---

## Loss

```
L_total = L_jepa + lambda_v * V(z_price) + lambda_c * C(z_price)
```
- `L_jepa = 1 - cosine_similarity(z_pred, z_target)` (or L2/d)
- Current defaults: `lambda_v = 10.0`, `lambda_c = 0.05` (see config.py -- the original canonical VICReg
  values of 25/1 assumed an 8192-dim expander WITH the invariance term; at d=256 without invariance,
  lambda_c=1.0 made covariance ~80% of the loss and suppressed the JEPA signal)
- Apply regularization to `z_price` only, not `z_pred` or `z_text`

If using codebook: replace `z_price` with `z_codebook` and add:
```
L_total += lambda_commit * ||z_price - sg(z_codebook)||^2
```

---

## Training Phases

| Phase | Steps | What is active |
|---|---|---|
| 1 | 1-500 | Projection layers only. Backbone + predictor adapters frozen. |
| 2 | 500+ | Projection layers + predictor adapters (LoRA/transformer body) at `lr_adapter`. Price encoder backbone at `lr_encoder` (separate param group). |
| 3 | Optional | Unfreeze top 2 price encoder layers at 0.1x lr if val loss plateaus. |

Each Phase-2 group ramps in via a warmup->cosine lr schedule (default: 200-step warmup, cosine to `lr_min_factor` * base by `max_steps`).

---

## Core Findings (as of 2026-06-05)

**Every attempt to train the price encoder backbone inside the JEPA loop diverges in BOTH train and val together.** Root cause: the EMA target encoder lags the moving backbone. When the backbone updates, the context representation shifts but the target representation (EMA copy) trails behind by ~1/(1-decay) steps. Context and target are already different time slices (unlike I-JEPA where they are two views of one input), so there is no "same input" anchor to prevent drift. Result: jepa loss rises monotonically as the predictor tries to track a moving target.

Evidence from training runs 1-4:
- Run 1: lr_encoder=3e-5 (shared) -> val_jepa plateaued at 0.092, encoder barely moved
- Run 2: lr_encoder=1e-4 -> val_jepa rose monotonically to 0.147 (encoder drifted too fast for EMA window of 500 steps)
- Run 3: lr_encoder=5e-5, ema_decay=0.999 -> first val_jepa decrease (0.0953->0.0945) at step 2000, but cosine schedule had already hit its floor
- Run 4-5: continued drift or premature early stopping

**Frozen-backbone runs are best so far (val_jepa 0.064)**, but suspected degenerate until Stage 0 diagnostics clear it. The suspicion: a frozen random backbone + trained projection head + EMA target encoder with the same projection head converge to similar mappings for any input window (EMA collapse), giving artificially low val_jepa without genuine temporal prediction.

**Stage 0 + Stage 1 are now DONE on nb03_best.pt (2026-06-05):** Stage 0 = degenerate (shuffled/true ratio 0.999); Stage 1 linear probe = no_signal AND anti-informative on val (trained z_price decodes future return/volatility/direction *worse* than an untrained random encoder — see the 2026-06-05 log). The frozen-random-backbone + trained-head + JEPA/VICReg loop is not merely empty, it is actively harmful on val.

**Next branch is gated on the backbone (pre-head) probe (added, pending a Colab run):**
- A pre-projection probe was added to nb03c (Test 4) and `linear_probe.py` (`probe_source="backbone"`). With `freeze_backbone=True` the `ProjectionHead` is the ONLY trained encoder module, so probing the pre-head `pooled` vs post-head `z_price` attributes the val degradation to head-vs-backbone.
- If **pooled R² ≫ z_price R²** (trained head destroys signal the frozen backbone kept): the loop/head/loss must change (reconstruction pretraining, drop/redesign the head, or train against pre-head features). **A pretrained backbone alone will NOT fix this.**
- If **pooled R² ≈ z_price R²** (both low): the head is fine; the frozen random backbone is simply a weak extractor → pivot to a frozen PRETRAINED backbone (TimesFM first — no numpy<2 dep; Moirai only in a clean numpy<2 env). Do NOT continue with a random frozen backbone.

---

## Open Decisions

1. **Anti-collapse regularization:** VICReg (V+C only) vs soft codebook bottleneck. Currently using VICReg. Add codebook if PCA/UMAP shows poor regime separation.

2. **Backbone quality:** The transformer backend is a from-scratch random initialization. No inductive bias from financial data. If Stage 0 says degenerate, the most likely fix is a pretrained backbone (Moirai or TimesFM), which requires a dedicated numpy<2 environment (not stock Colab). See 2026-06-04 log.

3. **Head vs. backbone (the open gate):** Stage 0 (degenerate) and Stage 1 (no_signal/anti-informative) are done on nb03_best.pt. The remaining question — is the trained head or the frozen random backbone responsible for the val anti-information? — is decided by the backbone (pre-head) probe added to nb03c Test 4. Run it on Colab; the result picks between "redesign the loop/head" and "pretrained backbone".

---

## Available ECC Tools (globally installed in ~/.claude/)

**Agents** (invoke via `Agent` tool with `subagent_type`):
- `mle-reviewer` -- ML code review: architecture, training loops, loss functions
- `code-reviewer` -- General code review
- `python-reviewer` -- Python/PEP 8/type hints
- `security-reviewer` -- Security audit
- `architect` -- System design decisions
- `planner` -- Task breakdown before implementation

**Skills most relevant here** (invoke via `/skill-name`):
- `/python-review` -- Python code review
- `/code-review` -- Review uncommitted changes

---

## File Structure

```
src/jepa_quant/
  config.py                    # All dataclasses + JEPAConfig; build_components() reads from here
  encoders/
    price_encoder.py           # TransformerPriceEncoder + MoiraiPriceEncoder; forward: [B,L,F] -> [B,D]
    projection.py              # ProjectionHead (Linear->GELU->Linear->LayerNorm); NEVER omit the LN
    target_encoder.py          # EMA wrapper; forward: [B,H,F] -> [B,D]; update() post-step only
    text_encoder.py            # FinBERT/Llama wrapper
  predictor/
    base.py                    # Predictor interface: forward(z_price, z_text) -> z_pred
    transformer_predictor.py   # Option A
    lm_predictor.py            # Option B (preferred)
  regularization/
    base.py                    # Regularizer interface + registry
    vicreg.py                  # V + C terms only (invariance intentionally dropped)
    codebook.py                # Soft codebook + commitment loss
  training/
    jepa_loss.py               # jepa_loss(z_pred, z_target, kind) -> scalar; 1 - cosine by default
    trainer.py                 # JEPATrainer + build_components + resolve_device
    ema.py                     # ema_update(target, source, decay)
    plot.py                    # Training history plotting utilities
  data/
    price_dataset.py           # PriceWindowDataset; pandas-free parquet reader; log-return features
    text_dataset.py
    conditioning_dataset.py
  eval/
    __init__.py
    diagnostics.py             # Stage 0: shuffled_target_control, compute_baselines, collapse_audit,
                               #   level_dependence_test, regime_clustering; all return dicts of numbers
    linear_probe.py            # Stage 1 (TBD): frozen encoder -> linear head -> downstream target
```

---

## Training Diagnostics to Log

- Codebook utilization (fraction of active vectors per batch) -- if <50%, reduce `lambda_commit`
- Per-dimension std of `z_price` -- variance collapse early warning
- JEPA loss vs regularization loss breakdown
- **Shuffled-target control at every val checkpoint** -- not just jepa loss

---

## Self-Improvement Protocol

After completing any non-trivial task, update this file if any of the following apply -- **only if every future agent in this codebase would need to know it**:

1. **Gotchas encountered**: silent failures, wrong assumptions about the architecture, hyperparameter interactions that weren't obvious
2. **Loops that required user intervention**: if you got stuck and the user had to redirect you, record what the trap was and how to avoid it
3. **Filter before writing**: ask "will every agent working in this codebase need to know this?" -- if no, do not add it
4. **Update Project Status below** to reflect the current phase, what's been decided, and what's still open

Use HTML files in `docs/` for any concept that benefits from a diagram. Reference them from this file.

---

## Project Status

> **Maintained by the self-improvement protocol. Update this after each task.**

- **Phase**: Stage 1 + nb03c Test 4 done on nb03_best.pt. Verdict: trained z_price is anti-informative on val AND nb03c Test 4 found TWO faults — (a) the trained projection head destroys signal its own frozen backbone preserved, (b) the frozen random backbone is itself weak (~0.08 vol R²). Conclusion: a pretrained backbone is NECESSARY BUT NOT SUFFICIENT. nb04 (no-JEPA raw-TimesFM CEILING probe) is built and pending a Colab run to gate the pivot.
- **Primary metric**: Downstream linear probe (Stage 1). val_jepa is secondary and must always be reported alongside its shuffled-target control ratio.
- **Open decision**: ceiling — run nb04 on Colab. Does RAW TimesFM (no JEPA/head/training) beat the untrained-random floor (+0.03) and raw inputs (+0.02) on future_volatility val R²? COMMIT → build a JEPA loop on a frozen TimesFM backbone; HOLD (C≈floor) → backbone swap won't help, reconsider targets/conditioning.
- **Decided**: Drop VICReg invariance term. Prefer Qwen2.5-1.5B + LoRA as predictor (Option B). Price encoder default = `transformer` backend. TimesFM loads via the HF `transformers` integration (`TimesFmModelForPrediction`), NOT the `timesfm` pip package (numpy<2 risk). No new training run until the nb04 ceiling probe gates the pivot.

### Log

**2026-06-06 -- nb04 no-JEPA raw-TimesFM CEILING probe built (DIAGNOSTIC); linear_probe refactored to encode_fn-based**
- Goal: before spending training units on a TimesFM pivot, measure whether RAW pretrained-backbone features linearly decode our three targets with NO JEPA / NO head / NO training — the ceiling any frozen-backbone JEPA inherits. Forward passes only.
- Refactored `src/jepa_quant/eval/linear_probe.py` to probe an arbitrary feature extractor without changing Stage 1 behavior: new `_collect_probe_pairs_fn(encode_fn, cfg, split, n_batches, device)` is the source-agnostic collector (target derivation + context re-normalization moved verbatim); `_collect_probe_pairs(components, …, probe_source)` is now a THIN wrapper that builds an encode_fn (z_price → `price_encoder`; backbone → the forward-pre-hook on `price_encoder.head`) and delegates. Fitting extracted to pure `_fit_regression`/`_fit_direction` (shared by both paths). New public `probe_regression_features` / `probe_direction_features` take an `encode_fn` directly. All linear-algebra primitives unchanged; default alpha grids unchanged (Stage 1 reproducibility). Bit-identical to the old components path — proven by `tests/test_linear_probe_refactor.py` (z_price-via-encode_fn == components path on the synthetic-parquet `cfg` fixture; 59/59 suite green).
- New `src/jepa_quant/eval/backbone_features.py`: `timesfm_encode_fn(checkpoint, device, *, channel=CLOSE_IDX, variant="close")` (RAW TimesFM pooled hidden states; `concat6` left as a documented NotImplementedError hook) and `raw_feature_encode_fn()` (per-channel [mean,std,last,sum] → [B,24] no-encoder baseline). Exported from `eval/__init__.py`.
- **TimesFM API decision (researched, not from memory):** use the HuggingFace `transformers` integration `TimesFmModelForPrediction.from_pretrained("google/timesfm-2.0-500m-pytorch")`, NOT the `timesfm` pip package. Rationale (mirrors the Moirai/torchao discipline): the pip package can pin numpy<2 and break Colab's numpy-2 stack; the transformers integration rides the already-installed numpy-2 `transformers` (needs `transformers>=4.48`, bumped from nb03c's 4.44). Embeddings = `outputs.last_hidden_state` `[B, n_patches, 1280]` mean-pooled over patches (NOT the forecast head). Config 2.0-500m: patch_length=32, hidden_size=1280, 50 layers; our context L=64 = exactly 2 patches; the ForPrediction wrapper patches/pads internally. `forward(past_values=<list of 1-D tensors>, freq=<long tensor>, return_dict=True)`. Import is lazy inside the factory (mirrors `MoiraiPriceEncoder`'s `uni2ts` import) so `import jepa_quant.eval.backbone_features` works without transformers/timesfm.
- `notebooks/nb04_backbone_ceiling.ipynb` (thin driver): full QUESTION/H1/H0/.../DECISION header (RESULT+DECISION blank); Colab setup (drive mount, git pull, `transformers>=4.48`, torchao uninstall); four encode_fns — raw-input, untrained random z_price (FLOOR), trained nb03_best z_price, RAW TimesFM (CEILING) — all scored by the IDENTICAL `probe_*_features` harness/seed/WIDE grid `(0.01…1e4)`. Wide grid because Stage 1 pinned best_alpha=100 (grid max) → under-regularized; nb04 re-runs floor/baseline under the wide grid and asserts no representation pins best_alpha on a grid boundary (widen if it does). 4×3 table + train-vs-val gap (TimesFM D=1280 overfit watch) + automated verdict (COMMIT if C>Fl+0.03 and C>Rw+0.02 on vol R²; HOLD if C≈Fl; drop direction if best acc≈majority).
- **Pending a Colab run** (data + TimesFM live on Drive). Verified locally only: ast.parse of every module + every non-magic nb04 cell; the bit-identical regression test; a synthetic-parquet end-to-end of the nb04 harness wiring (raw_input D=24 + z_price + wide grid + source_label + boundary fields all correct). Do NOT write a research_log entry until the user pastes Colab output (same as nb03c).

**2026-06-05 -- Stage 1 linear probe verdict = no_signal AND anti-informative; backbone (pre-head) probe added**
- Ran nb03c on nb03_best.pt (frozen random `transformer` backbone, d_model=768/n_layers=8, freeze_backbone=True; the trained head is the only encoder param). Probe = ridge / logistic on frozen z_price, trained vs untrained-random-encoder, same arch. n_val=2835 → binomial SE = sqrt(0.25/2835) = 0.0094, so the direction signal threshold 2·SE = 0.0188.
  - future_return    val R²: -0.069 (trained) vs -0.056 (untrained) → **Δval_r2_return = -0.013**
  - future_volatility val R²: +0.058 (trained) vs +0.097 (untrained) → **Δval_r2_vol = -0.039** (the untrained RANDOM projection decodes volatility BETTER)
  - direction        val acc: 0.528 vs 0.559 → **Δval_accuracy = -0.031** (|Δ| > 2·SE=0.0188); val AUROC 0.518 vs 0.569; trained acc 0.528 sits exactly on the majority-class baseline.
  - Train R² is comparable across trained/untrained (vol 0.186 vs 0.217) — the degradation is **val-only**.
- Conclusion: the frozen-random-backbone + trained-head + JEPA/VICReg loop is **anti-informative on val, not merely empty.** Training actively moves z_price away from downstream-decodable geometry on held-out data while train R² stays normal — a train/val generalization failure of the head, consistent with the Stage 0 degenerate (EMA-collapse) verdict.
- Since freeze_backbone=True makes the `ProjectionHead` the ONLY trained encoder module, the head is the only thing that could cause this. Added a **backbone (pre-projection) probe** to disambiguate head-vs-backbone: `probe_source="backbone"` in `src/jepa_quant/eval/linear_probe.py` (forward pre-hook on `price_encoder.head` captures `pooled`; backend-agnostic — Moirai also does pooled→head), threaded through `_collect_probe_pairs`/`linear_probe_regression`/`linear_probe_direction` and recorded in each returned dict. nb03c gained "Test 4" printing the 4-way table {trained,untrained}×{backbone pooled, z_price} on future_volatility (+ direction) and `head_destruction = backbone_val_r2 − zprice_val_r2` per model.
- **Pending a Colab run** (data + checkpoint live on Drive; not runnable here). Verified locally only: ast.parse of the module + every non-magic notebook cell, and a CPU smoke test that the pre-hook captures `pooled` [B, d_model] while the forward still returns z_price [B, latent_dim] and the two differ. Gate: pooled R²≫z_price R² → the trained head is the destroyer, a pretrained backbone alone won't help (change the loop/head/loss); pooled R²≈z_price R² → frozen random backbone is just weak, pivot to a pretrained backbone (TimesFM, no numpy<2 dep).

**2026-06-05 -- Stage 0 diagnostics module + research workflow introduced**
- Added `src/jepa_quant/eval/diagnostics.py` with five diagnostic functions: shuffled_target_control, compute_baselines, collapse_audit, level_dependence_test, regime_clustering.
- Added `notebooks/nb00_diagnostics.ipynb` as thin driver; all logic in modules.
- CLAUDE.md rewritten to mandate the QUESTION/H1/H0/MEASURED TARGET/DECISION RULE/PRIORS/FALSIFIER/RESULT/DECISION template for all notebooks.
- Primary metric changed from val_jepa to downstream linear probe (Stage 1, TBD).
- Documented core finding: every attempt to unfreeze the backbone in the JEPA loop diverges in both train and val. Frozen-backbone best so far but suspected degenerate.
- Mismatch corrected in CLAUDE.md: lambda_v=25/lambda_c=1 (old canonical values) replaced with actual defaults 10.0/0.05. File `train.py` corrected to `trainer.py`.

**2026-06-04 -- LM predictor (`backend='lm'`) blows up on Colab's preinstalled torchao 0.10.0**
- Symptom: `build_components(cfg)` with `USE_FOUNDATION_MODELS=True` crashes in `LMPredictor.__init__` at `get_peft_model(base, lora)` -> `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`. Not our code -- it's PEFT's LoRA dispatch.
- Root cause: Colab preinstalls `torchao==0.10.0`. Recent PEFT's `is_torchao_available()` **raises** (instead of returning `False`) when it finds a torchao below its 0.16.0 minimum. PEFT calls it from `dispatch_torchao` while building every LoRA layer, so the build dies even though we never use torchao quantization (plain LoRA on bf16 `Linear`).
- Fix (nb02 install cell, cell-6): add `%pip uninstall -q -y torchao` after the install. With torchao absent, `importlib.util.find_spec("torchao")` returns `None` -> `is_torchao_available()` returns `False` -> the torchao dispatcher is skipped cleanly. Uninstalling beats upgrading: bumping torchao to >=0.16.0 would drag torch/ABI churn onto Colab's numpy-2 stack.
- Operational gotcha: if you already ran the failed `build_components`, PEFT is imported into the kernel. After re-running the uninstall cell, do **Runtime -> Restart session** (the uninstall persists on disk across restart since the VM isn't recycled), then run from the config cell down -- guarantees no stale peft/torchao in `sys.modules`. A fresh top-to-bottom run (cell-6 before any peft import) doesn't need the restart.
- Rule: any agent enabling a PEFT/transformers backend on Colab should expect Colab's preinstalled ML libs (torchao, and historically torchvision/torchaudio) to be *older* than what current PEFT/transformers demand. Prefer removing the unused offender over version-bumping into ABI churn.

**2026-06-04 -- Switched price encoder default to `transformer`; Moirai shelved on Colab (ends the numpy saga)**
- Decision (user): after FOUR successive numpy failures on the numpy<2-for-Moirai path, switch `PriceEncoderConfig.backend` default `moirai -> transformer`. Root cause of the whole saga: `uni2ts` (Moirai) forces numpy<2, but Colab's entire prebuilt stack (torch, pandas, pyarrow, scipy, scikit-learn) is **numpy-2-built**. Forcing numpy 1.26 onto that stack is a losing battle -- each library trips a *different* numpy seam, so you get a new error after each fix.
- The transformer backend imports **no uni2ts/lightning/torchmetrics/scipy chain**, so the whole stack stays on Colab's native numpy 2 and all four errors vanish at once.
- Rule for any agent tempted to re-enable Moirai: **do NOT install uni2ts on Colab.** Run `backend='moirai'` only in a dedicated numpy<2 environment (local GPU / container) with the whole numeric stack pinned to its numpy-1-built releases.

**2026-06-03 -- Third ABI trap: torch.from_numpy rejects numpy-1 arrays (Colab torch is numpy-2-built)**
- Fix: `price_dataset._to_tensor()` builds the tensor from raw bytes -- `np.ascontiguousarray(...).tobytes()` (pure NumPy) -> `torch.frombuffer(bytearray(...), dtype=float32).reshape(shape)` (no NumPy ABI). Used for both `context` and `target`.

**2026-06-03 -- Training dataloader reads parquets pandas-free (two version traps)**
- `price_dataset._read_price_frame()` reads via `pyarrow.parquet.read_table` -> orders on the raw `ts` epoch in Arrow -> returns `{col: ndarray}` straight from `pyarrow .to_numpy()`. **Never calls `to_pandas`. Do not reintroduce pandas here.**

**2026-06-03 -- numpy<2 stack pinned for Moirai; notebooks install INLINE (not requirements.txt)**
- Gotcha (big one): **the notebooks do NOT `pip install -r requirements.txt`** -- each has its own inline `%pip install` list. Editing `requirements.txt` alone changes nothing on Colab. Pin versions in the **notebook install cell**.

**2026-06-02 -- Options/futures = conditioning, not training targets**
- Decided: options & futures are predictor *conditioning*, not training data. The JEPA target is the underlying price series. See full details in older CLAUDE.md log.

**2026-06-02 -- ECC setup**
- Gotcha: GateGuard has three distinct triggers per session, each requiring facts before retrying:
  1. **First Bash** -- state the user request and what the command produces
  2. **First Edit/Write per file** (including new files) -- state what imports it, no duplicate exists, data fields, user instruction
  3. **Destructive Bash** (git rm, reset, etc.) -- state files affected, one-line rollback, user instruction
  All gates pass on the second attempt after facts are presented.

**2026-06-02 -- Notebook env (local/VS Code vs Colab)**
- Gotcha: macOS python.org builds don't trust the system keychain, so `aiohttp` raises `SSL: CERTIFICATE_VERIFY_FAILED`. Fix is in `PolygonClient`: build `ssl.create_default_context(cafile=certifi.where())` and pass via `aiohttp.TCPConnector(ssl=ctx)`.
- Colab notebook sync: running the setup cell's `git pull` updates repo code but **cannot refresh the notebook tab you're viewing**. To get the latest notebook, reopen via File -> Open notebook -> GitHub tab.
- Gotcha: Colab's bundled IPython `autoreload` extension does `from imp import reload`, but `imp` was removed in Python 3.12. Fix: shim `sys.modules['imp']` with a minimal `types.ModuleType` that delegates `reload` to `importlib.reload` before calling `%load_ext autoreload`. Both notebooks have this shim.

**2026-06-03 -- Dev workflow (VS Code -> GitHub -> Colab)**
- Canonical loop: **edit `.py` modules in VS Code -> `git push` -> re-run setup cell in Colab (does `git pull`) -> re-run autoreload cell -> work cells pick up new code with no kernel restart.**
- Rule: logic lives in `src/jepa_quant/*.py`. Notebooks are thin drivers (imports + calls).
- Rule: **never edit code inside the Colab VM** (`/content/JEPA-quant`). Edits there are lost on VM recycle.
- Outputs write to `MyDrive/Colab Notebooks/JEPA-QUANT/data/...` via the mounted Drive.
