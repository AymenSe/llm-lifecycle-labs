# Project Status

> **Resume prompt for a new conversation:** "I'm continuing my LLM Lifecycle Labs project. Read STATUS.md, docs/ROADMAP.md and the current lab's README/REPORT, then review my latest evidence against that lab's acceptance criteria."

**Last updated:** 2026-10-21 · **Current lab:** Lab 0 (in progress, step 1: server inspection + GitHub repo) · **Current milestone:** M0 Foundations

## Context
- Owner: Aymen Sekhri. Mentor/reviewer: Claude (AI). Pace: ~4–6 h/week, alongside CME 295 (Fall 2026).
- Hardware: Linux server via SSH with 3× A100 PCIe 40 GB, **shared** (1–3 GPUs available at any time). Every lab therefore has a single-GPU baseline; multi-GPU parts are optional.
- Repo: public from day 1. URL: _TBD (add in Lab 0)_.

## Key decisions (see docs/ROADMAP.md §1 and docs/decisions/)
| Decision | Status |
|---|---|
| Architecture = SmolLM2-135M (Llama, 134,515,008 params), own PyTorch implementation with HF parity | agreed |
| Tokenizer = SmolLM2 (49,152 vocab), uint16 shards | agreed |
| Data = FineWeb-Edu `sample-10BT`, pinned revision | agreed |
| Token budget | **open**, decided in Lab 6 from measured throughput |
| Stack: Python 3.12, uv, torch 2.13.0 (cu126/cu130 chosen from the driver), transformers >=5.10.4,<5.18 | proposed (ADR 0001 in Lab 0) |
| Control model = SmolLM2-360M-Instruct (always labelled CONTROL) | agreed |
| PPO = own implementation (TRL removed PPOTrainer in 1.13) | agreed |
| vLLM optional (GRPO lab extension) | agreed |
| Tracking = MLflow, local SQLite store (no server); artifacts/checkpoints = HF Hub private repos + pinned revisions | agreed 2026-10-22 |
| New milestones: M4 MoE (from scratch, compute-matched vs dense) and M5 mid-training (GLM-5 inspired). **Lab numbering now follows docs/PLAN.md** | agreed 2026-10-22 |

## Lab tracker
| Lab | Title | Status | Evidence (PR / report) |
|---|---|---|---|
| 0 | Environment, hardware, Git/GitHub | not started | |
| 1 | Data inspection, provenance, filtering, dedup, decontamination | not started | |
| 2 | Tokenization, packing, shards | not started | |
| 3 | Transformer from scratch + HF parity | not started | |
| 4 | Causal LM objective + training loop | not started | |
| 5 | Optimization, BF16, accumulation, clipping | not started | |
| 6 | Profiling, throughput, compute budget | not started | |
| 7 | DDP + checkpoint/resume | not started | |
| 8 | Main pretraining run + base eval + KV cache | not started | |
| 9 | SFT + assistant-only masking | not started | |
| 10 | Full FT vs LoRA | not started | |
| 11 | Preference data + reward model | not started | |
| 12 | DPO | not started | |
| 13 | GRPO with verifiable rewards | not started | |
| 14 | PPO-RLHF (advanced; first to cut) | not started | |
| 15 | Final eval, inference demo, reproducibility audit, release | not started | |

Status values: `not started` → `in progress` → `in review` → `complete (verified: <what was checked>)`.

## Pinned facts verified on 2026-10-21 (re-check when you reach the relevant lab)
- torch 2.14.1 is the latest release; 2.13 has stable CUDA 12.6/13.0 builds. vLLM 0.31.0 pins torch==2.13.0, transformers <5.18.
- TRL 1.16.0: GRPO defaults `loss_type="dapo"`, `beta=0.0`, `scale_rewards="group"`; SFT `assistant_only_loss` needs `{% generation %}` template tags; PPOTrainer was removed in 1.13 (last available in 1.12.0 under `trl.experimental.ppo`).
- PEFT 0.21.2; lm_eval 0.4.13 (install as `lm_eval[hf]`); huggingface_hub 2.x (CLI is `hf`).

## Open questions / blockers
- Driver version on the server → CUDA index (Lab 0).
- Whether an NVLink bridge connects any GPU pair (Lab 0, `nvidia-smi topo -m`).

## Next actions
1. Create the GitHub repo + milestones; copy in the seed files.
2. Work through Lab 0 tasks 0.1–0.7 on branch `lab00/env-setup`.
3. Share the PR link + results for review.
