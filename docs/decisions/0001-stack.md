# ADR 0001 — Software stack and environment

- **Status:** accepted · **Date:** 2026-10-22 · **Lab:** 0

## Context
- Server XLIM-TESLA3: Ubuntu 24.04, 3× A100-PCIE-40GB, NVIDIA driver 560.35.05, which supports CUDA runtimes up to **12.6**.
- No NVLink. GPU0 is on NUMA 0; GPU1 and GPU2 are on NUMA 1 (NODE path). The GPUs are shared with other users.
- Later labs need TRL 1.16 and optionally vLLM. vLLM 0.31.0, the newest version TRL 1.16 accepts, pins `torch==2.13.0` and `transformers<5.18`.
- CI runs on CPU-only GitHub runners.

## Decision
- **Python 3.12**, managed by **uv**, with a committed `uv.lock`, resolved for Linux only.
- **torch 2.13.0** from the **cu126** PyTorch index. CI uses the CPU index. The two are declared as conflicting extras (`cpu`, `cu126`), and both indexes are `explicit = true`, so they serve only torch.
- transformers will be constrained to `>=5.10.4,<5.18` when it is added in Lab 3.
- Tracking: **MLflow** with a local SQLite store under `$LMLAB_ROOT` (no server). Artifacts and checkpoints go to **Hugging Face Hub** private repos, referenced by commit revision.
- Attention via PyTorch SDPA, without the flash-attn package.

## Alternatives considered
- **torch 2.14.1 (latest):** rejected. It is incompatible with the vLLM version TRL supports.
- **cu130 wheels:** impossible. They need a driver that supports CUDA 13; ours supports 12.6.
- **conda:** rejected. There is no cross-platform lockfile, and conda is slower and harder to reproduce in CI.
- **W&B:** replaced by MLflow, so no external account or data upload is needed.

## Consequences
- `uv sync --locked --extra cu126` reproduces the server environment; `--extra cpu` reproduces CI.
- Upgrading torch or CUDA requires a new ADR. If the driver is upgraded to R580+, cu130 becomes an option.
- **Risk:** vLLM wheels may target a newer CUDA than driver 560 supports. Re-check before the GRPO lab, and fall back to HF `generate` if needed.
- conda `(base)` must be deactivated in project shells.
