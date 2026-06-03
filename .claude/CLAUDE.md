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

**2026-06-03 — Price parquets need a version-proof reader (tz-aware index trap)**
- Gotcha: stock/option/futures parquets store a **tz-aware datetime index** (`ts: timestamp[ms, tz=UTC]`, written via `set_index('ts')`). On Colab, `requirements.txt` resolved to **pandas 2.1.4** against a newer bundled pyarrow, and rebuilding that index during `pd.read_parquet` crashes with `TypeError: datetime64 values must have a unit specified` (fails inside pyarrow `_reconstruct_index`/`_extract_index_level`). This stops `build_dataloaders` cold.
- Fix: `price_dataset._read_price_frame()` reads via `pyarrow.parquet.read_table` → orders on the raw `ts` epoch in Arrow (`cast(ts,int64)` + `sort_indices`) → `drop(['ts'])` → `to_pandas(ignore_metadata=True)`. Never reconstructs the datetime index. The dataset only needs price cols in chronological order, never the timestamps. **Do not revert to `pd.read_parquet(path).sort_index()`** — it's version-fragile.
- Note: the dep-conflict wall on Colab (pandas/numpy/scipy/torch pins fighting Colab's stack) is noisy but non-blocking; the autoreload `numpy`/`numpy.ma` errors are harmless. Only the datetime read was fatal.

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
