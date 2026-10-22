# Plan v2: MLflow, Hugging Face, MoE, GLM-5 lessons

*2026-10-22. Replaces the lab numbering in ROADMAP.md. The per-lab details in ROADMAP still apply unless changed here.*

## 1. Tooling

**MLflow: local store, no server**
- Install: `mlflow` (3.17.x), `psutil`, `nvidia-ml-py`.
- Point every run at one SQLite file on the data disk:
  `export MLFLOW_TRACKING_URI=sqlite:///$LMLAB_ROOT/mlflow/mlflow.db`
  Give each experiment an `artifact_location` under `$LMLAB_ROOT/mlflow/artifacts`.
- To view: run `mlflow ui --backend-store-uri $MLFLOW_TRACKING_URI --port 5000` on the server, and `ssh -L 5000:localhost:5000 <server>` from Windows.
- Log for each run:
  - **params:** the flattened resolved config;
  - **tags:** `git_sha`, `git_dirty`, `data_manifest`, `hf_repo`, `hf_revision`, `lab`;
  - **metrics:** per step;
  - **system metrics:** `mlflow.enable_system_metrics_logging()` (GPU utilization, memory, power);
  - **artifacts:** small files only (config, env.json, plots).
- HF Trainer: `report_to="mlflow"`. TRL should work the same way because its configs inherit from TrainingArguments, but this hasn't been verified yet; test it in the SFT lab.
- `mlflow.db` is not in Git. For each lab, export the runs that matter to `labs/*/results/*.csv` and plots, so GitHub readers can see the results without MLflow.

**Hugging Face Hub: artifacts**
- Checkpoints go to **private** model repos, e.g. `<you>/lmlab-135m-dense-base` and `<you>/lmlab-moe-16e4a`. Upload with `hf upload`.
- Record the repo + **commit revision** in `manifests/` and as an MLflow tag. Never refer to a checkpoint as just "latest".
- Datasets: pin the FineWeb-Edu revision. Uploading token shards to a dataset repo is optional.
- Release: at the end, make the final models public with model cards; add an optional Space demo.

## 2. Milestones and labs (E = essential, O = optional)

| M | Lab | Topic | New/changed |
|---|---|---|---|
| M0 | 0 | Env, hardware, Git/GitHub | MLflow replaces W&B |
| M1 Data | 1 | Inspection, licensing, filtering, dedup, decontamination | |
| | 2 | Tokenization, packing, shards | |
| M2 Model | 3 | Dense transformer + HF parity | |
| | 4 | LM loss + training loop | |
| | 5 | AdamW, schedules, BF16, accumulation, clipping | O: Muon (GLM-5 uses it) |
| M3 Dense base | 6 | Profiling, throughput → token budget | |
| | 7 | DDP + checkpoint/resume | |
| | 8 | **Dense pretraining run** + base eval | |
| **M4 MoE** | **9** | **MoE layer: router, top-k, load balancing (E)** | new |
| | **10** | **Compute-matched MoE vs dense run (E)** | new |
| **M5 Mid-training** | **11** | **Context extension + annealing mix (E)** | new, GLM-5 inspired |
| M6 SFT | 12 | SFT + assistant-only masking | + mask bad segments |
| | 13 | Full FT vs LoRA | |
| M7 Prefs | 14 | Preference data + reward model | |
| | 15 | DPO | |
| M8 RL | 16 | GRPO, verifiable rewards | + GLM-5 GRPO settings |
| | 17 | PPO-RLHF (O, first to cut) | |
| | 18 | On-policy distillation against forgetting (O) | new, GLM-5 §3.5 |
| M9 Release | 19 | Final eval, inference, reproducibility audit, release | |

**Pace:** at ~5 h/week, the essential path is now about **9–12 months**. To shorten it, cut 17, 18, then 13.

## 3. MoE milestone (Labs 9–10)

**Question:** with the same *active* parameters and the same tokens, does MoE beat the dense 135M model, and what does it cost?

- **Starting design:** fine-grained experts, so active FFN width matches dense.
  - Replace each FFN (width 1536) with **16 experts of width 384, top-4**. Active FFN width = 1536, the same as dense.
  - That is ~373M total params with ~135M active, which fits easily in 40 GB.
- **You implement:**
  - router (logits computed in fp32), softmax + top-k, renormalized weights;
  - token dispatch/combine (start with a loop over experts, then a sorted/grouped version);
  - **load-balancing aux loss** (Switch, α≈0.01) + **router z-loss** (≈0.001);
  - logging of per-expert load and dead experts.
- **Parity reference:** `MixtralForCausalLM` set to 16 experts / top-4 (no QK-norm, so it is closest to your dense code). Verify per-layer outputs match.
- **Ablations, one at a time:** with vs without shared expert · aux loss vs **aux-loss-free bias balancing** (2408.15664, what GLM-5/DeepSeek-V3 use) · top-2 vs top-4.
- **Fair comparison:** same data order, tokens, LR schedule and seed as the dense run. Report val loss **at equal tokens and at equal wall-clock**. A naive MoE is slower, and that counts.
- **Acceptance evidence:** parity test passes; expert-load histogram shows no dead experts; MoE vs dense loss curves; a throughput/memory table; a one-paragraph verdict.
- **Decision:** whichever base wins carries into post-training. The other stays as a documented baseline.
- **Read:** Switch (2101.03961), DeepSeekMoE (2401.06066), OLMoE (2409.02060) for small-scale MoE ablations, Loss-free balancing (2408.15664).

## 4. GLM-5 lessons and where they land

Source: *GLM-5: from Vibe Coding to Agentic Engineering*, arXiv 2602.15763 (744B total / 40B active MoE, 256 experts, top-8 + 1 shared).

| GLM-5 did | What you do at 135M | Lab |
|---|---|---|
| Mid-training: staged context extension 32K→128K→200K | Extend 2K→8K: change RoPE base, train on a few hundred M tokens of long docs, measure loss by position | 11 |
| Mid-training: upsample code/agent/synthetic data late | Anneal the LR to 0 on a high-quality mix (edu score ≥4 + task-format data you'll use in RL) | 11 |
| Aux-loss-free MoE balancing (sigmoid scores + bias) | Ablation in the MoE lab | 9–10 |
| SFT: mask erroneous segments out of the loss | Extend your loss mask to drop flagged turns; test it | 12 |
| Hybrid rewards (rules + outcome RM + generative RM) | Combine your programmatic checker with your RM; report each separately | 14, 16 |
| GRPO without KL, asymmetric clip ε_low 0.2 / ε_high 0.28, large groups | Try exactly this vs symmetric clip + KL; watch entropy | 16 |
| Training–inference mismatch fixes (IcePop, token-in/token-out) | Log rollout log-probs vs trainer log-probs and plot the gap (matters if you use vLLM) | 16 |
| MoE "routing replay" in RL | Only if you RL the MoE model: check routing agreement between rollout and update | 16 (O) |
| On-policy cross-stage distillation to undo forgetting | Distill from earlier-stage checkpoints after RL; measure recovered skills | 18 (O) |
| Async RL infra (slime), DSA sparse attention, MTP, FP8 | **Read only**: not needed at this scale | — |

## 5. What to read, in order
1. CME 295 L1–L5 (alongside Labs 3–8).
2. Switch → DeepSeekMoE → OLMoE → Loss-free balancing (before Lab 9).
3. GLM-5 report: §2.3 mid-training (before Lab 11); §3.1–3.5 post-training (before Labs 12–18).
4. GLM-4.5 (2508.06471) §post-training: expert models + self-distillation (optional context).
