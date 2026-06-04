# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **Cost is not a concern.** The user is on a monthly (flat-rate) plan, so session cost warnings are irrelevant here — do not factor spend into decisions, suggest cheaper shortcuts to save money, or wrap up early to limit cost. Optimize purely for correctness and getting the task done well.

## Architecture

Three-component JEPA system. See [`docs/architecture.html`](docs/architecture.html) for the data flow diagram.

| Component | Frozen? | Notes |
|---|---|---|
| Price Encoder (context) | Yes (initially) | **Default backend = `transformer`** (lightweight, native numpy 2). Moirai is architecturally preferred but its `uni2ts` dep forces numpy<2 and is unusable on stock Colab — only run `backend='moirai'` in a clean numpy<2 env. Unfreeze top 2 layers only if domain shift confirmed |
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
- **Decided**: Drop VICReg invariance term (conflicts with JEPA temporal objective). Prefer Qwen2.5-1.5B + LoRA as predictor (Option B). **Price encoder default = `transformer` backend** (Moirai shelved on Colab — see 2026-06-04 log).

### Log

**2026-06-04 — LM predictor (`backend='lm'`) blows up on Colab's preinstalled torchao 0.10.0**
- Symptom: `build_components(cfg)` with `USE_FOUNDATION_MODELS=True` crashes in `LMPredictor.__init__` at `get_peft_model(base, lora)` → `ImportError: Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`. Not our code — it's PEFT's LoRA dispatch.
- Root cause: Colab preinstalls `torchao==0.10.0`. Recent PEFT's `is_torchao_available()` **raises** (instead of returning `False`) when it finds a torchao below its 0.16.0 minimum. PEFT calls it from `dispatch_torchao` while building every LoRA layer, so the build dies even though we never use torchao quantization (plain LoRA on bf16 `Linear`).
- Fix (nb02 install cell, cell-6): add `%pip uninstall -q -y torchao` after the install. With torchao absent, `importlib.util.find_spec("torchao")` returns `None` → `is_torchao_available()` returns `False` → the torchao dispatcher is skipped cleanly. Uninstalling beats upgrading: bumping torchao to ≥0.16.0 would drag torch/ABI churn onto Colab's numpy-2 stack.
- Operational gotcha: if you already ran the failed `build_components`, PEFT is imported into the kernel. After re-running the uninstall cell, do **Runtime → Restart session** (the uninstall persists on disk across restart since the VM isn't recycled), then run from the config cell down — guarantees no stale peft/torchao in `sys.modules`. A fresh top-to-bottom run (cell-6 before any peft import) doesn't need the restart.
- Rule: any agent enabling a PEFT/transformers backend on Colab should expect Colab's preinstalled ML libs (torchao, and historically torchvision/torchaudio) to be *older* than what current PEFT/transformers demand. Prefer removing the unused offender over version-bumping into ABI churn.

**2026-06-04 — Switched price encoder default to `transformer`; Moirai shelved on Colab (ends the numpy saga)**
- Decision (user): after FOUR successive numpy failures on the numpy<2-for-Moirai path, switch `PriceEncoderConfig.backend` default `moirai → transformer`. Root cause of the whole saga: `uni2ts` (Moirai) forces numpy<2, but Colab's entire prebuilt stack (torch, pandas, pyarrow, scipy, scikit-learn) is **numpy-2-built**. Forcing numpy 1.26 onto that stack is a losing battle — each library trips a *different* numpy seam, so you get a new error after each fix: (1) `datetime64 unit` (pyarrow→pandas), (2) `expected numpy.ndarray, got numpy.ndarray` (pandas make_block C-ABI), (3) same at torch↔numpy (`from_numpy`), (4) `numpy.linalg has no attribute _umath_linalg` (autoreload + half-applied downgrade corrupting numpy's C submodules). "Same root, different errors" = the rot surfaces wherever the next library happens to touch numpy.
- The transformer backend imports **no uni2ts/lightning/torchmetrics/scipy chain**, so the whole stack stays on Colab's native numpy 2 and all four errors vanish at once. Verified: default `PriceEncoderConfig()` builds `TransformerPriceEncoder`, `uni2ts` never imported; 56/56 tests pass.
- Changes: `config.py` default flipped (with rationale docstring); nb02 install cell now `%pip install transformers peft accelerate einops matplotlib pyarrow` (no pins, no uni2ts, no restart needed); `requirements.txt` unpinned to numpy 2, uni2ts moved to an OPTIONAL commented block.
- Rule for any agent tempted to re-enable Moirai: **do NOT install uni2ts on Colab.** Run `backend='moirai'` only in a dedicated numpy<2 environment (local GPU / container) with the whole numeric stack pinned to its numpy-1-built releases.
- The two earlier code workarounds (`price_dataset` pandas-free reader + `_to_tensor` frombuffer) are kept: they're version-agnostic (correct under both numpy 1 and 2) and the pandas-free reader still dodges the independent pyarrow→pandas datetime64 bug.
- Gotcha that prolonged the saga: nb02's autoreload cell is `%autoreload 2` with NO exclusions, so a fresh `pip install` of numpy rewrites its mtimes and autoreload re-executes numpy's .py files but can't re-bind its C extensions → corrupted numpy (`_umath_linalg`/`_NoValue`). Not an issue on numpy 2 now (no downgrade), but if you ever pin a compiled lib, exclude it from autoreload (`%aimport -numpy -scipy -torch`) or restart cleanly.

**2026-06-03 — Third ABI trap: torch.from_numpy rejects numpy-1 arrays (Colab torch is numpy-2-built)**
- Symptom: `PriceWindowDataset.__getitem__` crashed in the DataLoader worker with `TypeError: expected np.ndarray (got numpy.ndarray)` at `torch.from_numpy(np.ascontiguousarray(...))`. Same C-ABI family as the pandas/pyarrow traps, but at the **torch↔numpy** boundary: Colab's preinstalled (CUDA) torch wheel is built against **numpy 2** while uni2ts pins numpy to 1.26, so `from_numpy` refuses the numpy-1 array (identical repr, different C type identity).
- Fix (no torch reinstall): `price_dataset._to_tensor()` builds the tensor from raw bytes — `np.ascontiguousarray(...).tobytes()` (pure NumPy) → `torch.frombuffer(bytearray(...), dtype=float32).reshape(shape)` (no NumPy). Neither side's ABI is exercised. `bytearray` dodges torch's non-writable-buffer warning. Used for both `context` and `target`.
- Why enough: it's the **only** torch↔numpy crossing in the hot path — once samples are tensors, the rest (encoders, loss, EMA) is pure torch. Codebook k-means init (sklearn) is the next likely numpy-built wall, but it's Phase-2+/optional.
- Escape hatch unchanged: if ABI walls keep piling up, `backend='transformer'` drops uni2ts and keeps the whole Colab stack on its native numpy 2 (consistent, no FM). The numpy<2 path means patching each numpy-2-built C-extension boundary one at a time.

**2026-06-03 — Training dataloader reads parquets pandas-free (two version traps)**
- `price_dataset._read_price_frame()` reads via `pyarrow.parquet.read_table` → orders on the raw `ts` epoch in Arrow (`cast(ts,int64)` + `sort_indices`) → `drop(['ts'])` → returns `{col: ndarray}` straight from `pyarrow .to_numpy()`. **It never calls `to_pandas`/`pd.read_parquet`, never constructs a DataFrame, and the module no longer imports pandas.** `_timestep_features` accepts any `Mapping[str, ndarray]` (coerces via `np.asarray`). Do not reintroduce pandas here.
- Trap 1 (tz-aware index): parquets store a tz-aware datetime index (`ts: timestamp[ms, tz=UTC]`, via `set_index('ts')`). Letting pandas rebuild it during `pd.read_parquet` crashes with `TypeError: datetime64 values must have a unit specified` (pyarrow `_reconstruct_index`) when pandas (2.1.x) is older than the bundled pyarrow.
- Trap 2 (NumPy C-ABI): `uni2ts` requires `numpy<2`, so `%pip install -r requirements.txt` downgrades Colab's NumPy to 1.26.4 — but Colab's prebuilt **pandas/pyarrow wheels are built against NumPy 2**. Handing NumPy-1 arrays to those C extensions raises `TypeError: Argument 'values' has incorrect type (expected numpy.ndarray, got numpy.ndarray)` inside pandas `make_block`. **pyarrow->NumPy stays consistent** (pyarrow produced the array pandas rejected), which is why the pandas-free path works. This ABI skew still breaks *any* other pandas use on that Colab session (e.g. the ingestion notebook) — the real cure is keeping NumPy 2 (don't install `uni2ts` unless using the Moirai backend; it's optional).
- Note: the dep-conflict wall and `autoreload of numpy`/`numpy.ma` errors on Colab are noisy but non-blocking.

**2026-06-03 — numpy<2 stack pinned for Moirai; notebooks install INLINE (not requirements.txt)**
- Gotcha (big one): **the notebooks do NOT `pip install -r requirements.txt`** — each has its own inline `%pip install` list (nb 02 cell 5, nb 01). Editing `requirements.txt` alone changes nothing on Colab. Pin versions in the **notebook install cell** (and keep `requirements.txt` in sync for the local `.venv`).
- Root cause of the whole 2026-06-03 error chain: default `price_encoder.backend='moirai'` → `uni2ts` → requires **numpy<2**, which downgrades Colab's numpy to 1.26.4, but `pyarrow`/`scikit-learn` (left unpinned) stayed at Colab's **numpy-2-built** wheels. numpy-2-built C extensions can't accept numpy-1 arrays → the datetime64-unit, make_block, and `_NoValueType` errors.
- Resolution (decided): keep Moirai, pin the **entire compiled numeric stack** to its last numpy-1-built releases so they agree: `numpy==1.26.4, pandas==2.1.4, pyarrow==15.0.2, scipy==1.11.4, scikit-learn==1.4.2` (torch left to uni2ts, which pins ~2.4.x). After installing, **Runtime → Restart** so the numpy-1 wheels load before anything imports numpy (also clears the autoreload-corrupted numpy state).
- Alternative if you ever drop Moirai: set `backend='transformer'`, remove `uni2ts` + all the numpy pins, keep Colab's numpy 2 (simpler, no FM).

**2026-06-02 — Options/futures = conditioning, not training targets**
- Decided: options & futures are predictor *conditioning* (a few daily features per underlying — ATM IV, skew, term slope), not training data. The JEPA target is the underlying price series. So per-contract OHLCV history is collected only to *derive* features, never fed raw.
- Gotcha: historical option features require **expired-contract enumeration** (`/v3/reference/options/contracts?expired=true`), NOT the current chain — current-chain contracts haven't existed long enough to carry history. Notebook now enumerates expired contracts, keeps ATM±N strikes at ~monthly expiries, caps at `OPT_MAX_CONTRACTS`/`OPT_MAX_RUNTIME_H`.
- Gotcha: Polygon free tier is **5 req/min per API key (account-wide), not per endpoint** — async/concurrency cannot beat it. Only fewer requests, a paid tier, or more keys help.
- Plan: train conditioning path with **conditioning dropout (30–50%)** so it learns from the options-covered subset and degrades gracefully when absent; derive IV via Black–Scholes inversion (greeks/IV are paid).
- Gotcha (corrected): futures are NOT `ES1!` continuous symbols on `/v2/aggs` (always empty). They live on the **dedicated Futures API** (free *Futures Basic* tier): enumerate dated single contracts via `/futures/v1/contracts?product_code=ES&type=single` then pull daily bars via `/futures/v1/aggs/{ticker}?resolution=1session` with `window_start.gte/lte` (ns-epoch `window_start`, paginates on `next_url`, no vwap → derive from `dollar_volume/volume`). Symbology = product code + CME month letter + year digit (`ESU5`).
- Gotcha: `/futures/v1/contracts` only accepts **`sort` columns `{date, product_code, ticker}`** (dotted `.asc`/`.desc` direction). `last_trade_date` is *filterable* (`last_trade_date.gte`) but **NOT sortable** — `sort=last_trade_date.desc` returns `400 Invalid query parameter: 'sort'`. Use `sort=date.desc` (most-recently-active first). Note `client.get`'s `raise_for_status()` discards Polygon's JSON error body, so the actual reason is invisible from the traceback — reproduce the bare URL with `curl -H "Authorization: Bearer $KEY"` to read the `error` field. Polygon's futures docs now redirect to `massive.com`; the aggs endpoint's `sort=window_start.asc` is fine.

**2026-06-02 — ECC setup**
- Gotcha: GateGuard has three distinct triggers per session, each requiring facts before retrying:
  1. **First Bash** — state the user request and what the command produces
  2. **First Edit/Write per file** (including new files) — state what imports it, no duplicate exists, data fields, user instruction
  3. **Destructive Bash** (git rm, reset, etc.) — state files affected, one-line rollback, user instruction
  All gates pass on the second attempt after facts are presented.

**2026-06-02 — Notebook env (local/VS Code vs Colab)**
- Gotcha: macOS python.org builds don't trust the system keychain, so `aiohttp` raises `SSL: CERTIFICATE_VERIFY_FAILED` against `api.polygon.io`. Fix is in `PolygonClient`: build `ssl.create_default_context(cafile=certifi.where())` and pass it via `aiohttp.TCPConnector(ssl=ctx)`. Portable — no-op on Colab/Linux. `certifi` is in `requirements.txt`.
- Colab notebook sync: running the setup cell's `git pull` updates repo code and any Drive copy, but **cannot refresh the notebook tab you're viewing** (a cell can't reload its own document). To get the latest notebook, reopen via File → Open notebook → GitHub tab. The old `shutil.copy2`-to-Drive block was removed because Colab autosave races it and clobbers the synced file.
- Setup cell guards: `IN_VSCODE = VSCODE_PID/VSCODE_CWD present` forces `IN_COLAB=False` so a local VS Code kernel never triggers the Drive mount / clone. Limitation: a *remote* Colab kernel driven from VS Code won't expose `VSCODE_PID`, so it's still treated as Colab.
- Local dev: use `.venv` (gitignored) as the VS Code kernel; deps in `requirements.txt`. The notebook's `%pip install` cell is then a fast no-op.
- Gotcha: Colab's bundled IPython `autoreload` extension does `from imp import reload`, but `imp` was removed in Python 3.12. Fix: shim `sys.modules['imp']` with a minimal `types.ModuleType` that delegates `reload` to `importlib.reload` before calling `%load_ext autoreload`. Both notebooks have this shim in their autoreload cell.

**2026-06-03 — Dev workflow (VS Code → GitHub → Colab)**
- Canonical loop: **edit `.py` modules in VS Code → `git push` → re-run setup cell in Colab (does `git pull`) → re-run autoreload cell → work cells pick up new code with no kernel restart.**
- Rule: logic lives in `src/jepa_quant/*.py`. Notebooks are thin drivers (imports + calls). Keeping notebooks thin means you almost never need to reopen from the GitHub tab — only module changes flow through the loop.
- Rule: **never edit code inside the Colab VM** (`/content/JEPA-quant`). Edits there are lost on VM recycle and will conflict on the next `git pull`. Edit on the Mac, push, pull — one direction only.
- Rule: **never click "Copy to Drive"** on the Colab banner. Notebooks live on GitHub on purpose; a Drive copy drifts from GitHub silently.
- Notebooks: open both from **File → Open notebook → GitHub tab**, not from Drive. "Copy to Drive" banner = healthy state, not an error.
- Outputs (data / checkpoints / plots) write to `MyDrive/Colab Notebooks/JEPA-QUANT/data/...` via the mounted Drive. To get them on the Mac, install Google Drive for Desktop — the folder syncs automatically, no git involved.
- Notebook edits made in Colab (cell additions etc.): use **File → Save a copy in GitHub** to commit back, then `git pull` locally. Do not use "Save to Drive" for this.
- Do not use the Colab-in-VS-Code tunnel/extension. It adds a fragile tunnel but does not remove the git push/pull loop — code in VS Code still runs against the VM's git clone, not the local file you're viewing.
