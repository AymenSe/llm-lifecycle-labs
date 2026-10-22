# LLM Lifecycle Labs: Roadmap

A lab-based project that takes **one small decoder-only model** from random initialization through pretraining, SFT, LoRA, reward modeling, DPO, GRPO and PPO. Every claim in the final portfolio must point to code, tests, logs or reports in this repository.

*Last verified against upstream releases: 2026-10-21. Version facts below were checked on PyPI, GitHub release notes and official docs; items marked (unverified) still need checking on your server.*

---

## 1. Decisions on the proposal

| Proposal | Verdict | Notes |
|---|---|---|
| ~135M decoder-only, SmolLM2-135M architecture | **Keep** | Plain `LlamaForCausalLM`: 30 layers, d=576, FFN 1536, 9 query heads / 3 KV heads (GQA, head_dim 64), RoPE θ=1e5, RMSNorm, SwiGLU, tied embeddings. Exact count: **134.5M params** (106.2M non-embedding + 28.3M embedding). It is small enough to iterate on one A100 and fully HF/vLLM-compatible. The only newer HF small model (nanowhale-100m: MLA+MoE, `trust_remote_code`, fp32 only) is a worse template. |
| Reuse SmolLM2 tokenizer | **Keep** | 49,152-vocab byte-level BPE. It already contains `<|im_start|>`/`<|im_end|>`, so SFT needs no new embeddings, and ids fit in **uint16**, which halves shard size. Training your own BPE is an optional stretch in Lab 2. |
| FineWeb-Edu | **Keep, use `sample-10BT`** | ODC-By 1.0 + CommonCrawl ToU. About 28.5 GB parquet and ~10B tokens, which is more than any budget we can afford. Pin the dataset **revision hash**. |
| Main pretraining run | **Budget decided in Lab 6** | Planning prior only, **not a promise**: 6N+attention ≈ 1.23 GFLOP/token at seq 2048. At 25–40% MFU on a 312 TFLOPS A100 that gives **~60–100k tokens/s/GPU**. Chinchilla-style 20 tokens/param ≈ 2.7B tokens is the floor; small models keep improving well past that (SmolLM2-135M saw 2T). Your measured throughput sets the real number. |
| Write own model vs. use HF | **Both** | You implement the transformer in PyTorch with HF-compatible parameter names, prove logit parity against `transformers.LlamaForCausalLM`, then train your own code and export to HF format for post-training. |
| Control model | **SmolLM2-360M-Instruct** (Apache-2.0) | Same tokenizer and chat format, so experiments can switch models with one config line. Always labelled `CONTROL` in reports. |
| TRL `PPOTrainer` | **Removed in TRL 1.13** | You implement PPO yourself in Lab 14, following *The N+ Implementation Details of RLHF with PPO* (arXiv 2403.17031). Optional cross-check: `trl==1.12.0` `trl.experimental.ppo` in an isolated environment. |
| vLLM | **Optional (Lab 13 extension)** | HF `generate` is fast enough for 135M. vLLM 0.31.0 is the highest version TRL 1.16 accepts, and it pins `torch==2.13.0`, which is why we pin torch 2.13 now. |
| flash-attn package | **Skip** | PyTorch SDPA's flash backend supports A100/BF16. The flash-attn package is sdist-only and must be compiled. |
| DeepSpeed / Megatron / nanotron / Ray / Docker / DVC | **Not needed** | Plain DDP fits a 135M model (~2 GB of train state per GPU). FSDP2 is an optional teaching extension. HF Hub + manifests replace DVC. |

## 2. Overall architecture

```
 FineWeb-Edu (pinned rev) ──► inspect/filter/dedup/decontam ──► tokenize+pack ──► uint16 shards + manifest(sha256)
                                                                                         │
                     your transformer (src/lmlab/model) ◄── parity test ── HF LlamaForCausalLM
                                         │
          single-GPU loop ──► BF16/AdamW/schedule ──► profiling/budget ──► DDP + exact resume
                                         │
                                 MAIN PRETRAIN RUN ──► base eval (val ppl, lm-eval, samples) ──► HF export
                                         │
          SFT (your loop, then TRL as a cross-check) ──► Full FT vs LoRA
                                         │
          preference data (human / AI / programmatic) ──► reward model ──► DPO
                                         │
          GRPO on verifiable procedural tasks ──► PPO-RLHF (policy, ref, RM, value)
                                         │
          final eval + KV-cache inference demo + reproducibility audit + model/data cards + release
```

Every training script follows one contract: **config YAML in, run directory out**. A run directory holds `config.resolved.yaml`, `env.json`, `git.json` (sha + dirty flag), `data_manifest.json`, `metrics.jsonl` and checkpoints. Plots and reports are generated from `metrics.jsonl`, never from screenshots. W&B is a mirror, not the source of truth.

## 3. Repository structure (target, grown lab by lab)

```
llm-lifecycle-labs/
├── README.md                  # what this is, results table, how to reproduce
├── STATUS.md                  # resume point for any new conversation
├── AI_ASSISTANCE.md           # transparent log of AI help
├── pyproject.toml  uv.lock  .python-version
├── .pre-commit-config.yaml  .gitignore
├── .github/
│   ├── workflows/ci.yml       # CPU: ruff + pytest (tiny configs)
│   ├── ISSUE_TEMPLATE/lab.md
│   └── pull_request_template.md
├── configs/{env,data,model,pretrain,sft,lora,rm,dpo,grpo,ppo,eval}/*.yaml
├── src/lmlab/
│   ├── utils/      env.py bench.py seed.py io.py provenance.py logging.py
│   ├── data/       inspect.py filters.py dedup.py decontam.py tokenize.py shards.py loader.py
│   ├── model/      config.py layers.py transformer.py hf_io.py
│   ├── train/      loop.py optim.py schedule.py precision.py checkpoint.py distributed.py profiler.py
│   ├── eval/       perplexity.py harness.py samples.py
│   ├── infer/      generate.py kv_cache.py
│   └── post/       chat.py masking.py sft.py lora.py prefs.py reward_model.py dpo.py
│                   grpo.py ppo.py rewards/{arith.py,format.py,...}
├── scripts/                   # thin CLIs: python scripts/pretrain.py --config ...
├── tests/                     # CPU-fast; GPU tests marked @pytest.mark.gpu
├── labs/labNN-<slug>/
│   ├── README.md              # instructions (from mentor)
│   ├── REPORT.md              # your lab report
│   └── results/               # small JSON/CSV/PNG only (<1 MB each)
├── manifests/                 # data shards, checkpoints, HF revisions, sha256
└── docs/
    ├── ROADMAP.md  reproducibility.md  data_card.md  model_card.md
    ├── evaluation_report.md  skills_evidence.md
    └── decisions/NNNN-*.md    # short ADRs (stack, tokenizer, budget, ...)
```

**Outside Git** (server data disk, root in `$LMLAB_ROOT`): raw data, shards, run directories, checkpoints, HF/uv caches. **Referenced from Git** through `manifests/*.yaml` entries (HF repo id + commit revision, or path + sha256, + producing git sha + config hash). Milestone checkpoints go to **private HF Hub repos**; the final ones are made public at release.

## 4. Software stack (pins decided in Lab 0, exact versions in `uv.lock`)

| Layer | Choice | Constraint / reason |
|---|---|---|
| Python | 3.12 | TRL needs ≥3.11; vLLM supports 3.11–3.14 |
| Env manager | uv (0.13.x) | lockfile, explicit PyTorch CUDA index, CPU extra for CI |
| PyTorch | **2.13.0**, `cu126` or `cu130` index chosen from the driver | 2.14.1 is newest, but vLLM 0.31 (TRL's max) pins 2.13.0 |
| HF core | transformers `>=5.10.4,<5.18`, datasets 5.x, tokenizers (from transformers), huggingface_hub (`hf` CLI), safetensors | v5 breaking changes: `rope_parameters` replaces `rope_theta`, `apply_chat_template` returns a dict, `use_cache` default False |
| Post-training reference | TRL 1.16.x (SFT, Reward, DPO, GRPO, RLOO) | used to **cross-check** your implementations, not replace them |
| PEFT | 0.21.x | LoRA / rsLoRA / DoRA |
| Eval | `lm_eval[hf]` 0.4.13 | lighteval has had no release for ~11 months |
| Tracking | local `metrics.jsonl` + W&B (free academic) or trackio | the JSONL is canonical |
| Quality | ruff, pytest, pre-commit (large-file + private-key hooks) | |
| Optional | vLLM 0.31.0, datasketch (MinHash), FSDP2 | only in the labs that need them |

## 5. Labs and milestones

Legend: **[E]** essential, **[O]** optional extension. Effort is *your* hours, uncertainty about ±50%. GPU-h are A100-hours, as rough priors to be replaced by measurements. CME 295 Fall 2026 alignment in brackets.

At ~5 h/week, the essential path is about **120–170 h, roughly 6–9 months**. If you need to shorten it, Lab 14 (PPO) and the depth of Lab 10 are the first things to cut.

### M0 — Foundations
**Exit criteria:** fresh clone → `uv sync --locked` → env report + GPU benchmark reproduced; CI green on PR; secrets/large-file hooks proven to block.

**Lab 0 — Environment, hardware, Git/GitHub [E]** · 5–8 h · <0.5 GPU-h
Question: *Can anyone rebuild my exact environment, and what can my hardware really deliver?* Full instructions in `labs/lab00-setup/README.md`.

### M1 — Data
**Exit criteria:** versioned, decontaminated, documented token shards with sha256 manifest; a data card; a loader that resumes deterministically.

**Lab 1 — Data inspection, provenance, filtering, dedup, contamination [E]** · 8–12 h · CPU
- Question: *What exactly will my model read, am I allowed to use it, and is my eval data leaking into it?*
- You implement: streaming inspection stats (length, language-ID score, edu-score distribution, URL/domain mix); heuristic filters with per-filter drop counts; exact dedup (normalized hash); n-gram (13-gram) decontamination against the eval sets you'll use (HellaSwag, ARC, PIQA, and later your SFT/RL test splits).
- Libraries: `datasets` streaming, pyarrow; `datasketch` allowed for MinHash in the stretch.
- Evidence: filter-funnel table, 50 hand-labelled kept/dropped examples, contamination hit report, `docs/data_card.md` with license chain.
- Stretch [O]: MinHash near-dedup and its precision on a labelled sample.

**Lab 2 — Tokenization, boundaries, packing, shards [E]** · 6–10 h · CPU
- Question: *How do documents become reproducible training sequences without leaking across boundaries or wasting tokens?*
- You implement: parallel tokenization to uint16 shards with EOS separators; a packing strategy (concat-and-chunk vs. document-aware) and its attention/position implications; a train/val split by document hash; a deterministic, resumable, rank-aware loader.
- Evidence: tokens/byte stats, a round-trip decode test, packing-efficiency numbers, a loader resume test (identical batch stream after restart), `manifests/data_v1.yaml`.
- Stretch [O]: train a 49k BPE on your own subset and compare compression with SmolLM2's tokenizer.

### M2 — Model and training loop  [CME 295 L1–L3]
**Exit criteria:** your model matches HF Llama logits; the loop overfits a batch; an LR sweep produces explained curves.

**Lab 3 — Decoder-only transformer from scratch [E]** · 8–12 h · CPU + minutes of GPU
- Question: *Is my transformer numerically the same model as the reference architecture?*
- You implement: RMSNorm, RoPE, GQA attention via `F.scaled_dot_product_attention(is_causal=True)`, SwiGLU MLP, tied embeddings, init scheme, HF-name-compatible state dict, `from_hf`/`to_hf` (v5 `rope_parameters`).
- Libraries: `transformers` only as the **reference** in tests.
- Evidence: parity test (max-abs logit diff < 1e-4 fp32 on random weights), exact param count 134,515,008, causality test (future tokens don't change past logits), RoPE relative-position test.
- Stretch [O]: a manual attention path with an explicit mask, compared against SDPA.

**Lab 4 — Causal LM objective and minimal training loop [E]** · 5–8 h · ~1 GPU-h
- Question: *Does my loss compute what I think it computes, and can the loop learn?*
- You implement: shifted cross-entropy with ignore-index, a single-GPU loop, eval perplexity, basic sampling (temperature/top-k/top-p), a JSONL logger, run-directory provenance.
- Evidence: initial loss ≈ ln(49152) ≈ 10.80; a single batch overfits to <0.1 loss; a tiny-model run on real shards with train/val curves.

**Lab 5 — Optimization: AdamW, schedules, BF16, accumulation, clipping [E]** · 6–10 h · 4–8 GPU-h
- Question: *Which optimization choices matter at this scale, and how do I know?*
- You implement: param groups (no weight decay on norms/embeddings, justified), warmup + cosine and WSD schedules, bf16 autocast, grad accumulation equivalence, global-norm clipping, logging of grad-norm, update/param ratio and loss spikes.
- Evidence: accumulation test (k×micro == 1×big within tolerance), LR sweep of 4–6 values on the debug model, fp32 vs bf16 curves, a documented spike/divergence case.

### M3 — Scaling the run  [CME 295 L5]
**Exit criteria:** measured throughput → budget ADR; DDP with bitwise-consistent resume; base model trained, evaluated, exported.

**Lab 6 — Profiling, throughput, memory, compute budget [E]** · 5–8 h · 2–4 GPU-h
- Question: *Where does time and memory go, and what token budget can I afford?*
- You implement: a tokens/s + MFU meter, `torch.profiler` traces, memory snapshots, comparisons of eager vs `torch.compile`, batch/seq sweeps, and activation checkpointing on/off.
- Evidence: an MFU table, a memory breakdown vs. analytical estimate, and `docs/decisions/000X-token-budget.md` with the calculation hours = tokens / (tok/s × GPUs × 3600) and a sharing-aware calendar.

**Lab 7 — DDP and correct checkpoint/resume [E]** · 8–12 h · 3–6 GPU-h
- Question: *Does 3-GPU training compute the same thing as 1-GPU, and can it survive interruption?*
- You implement: `torchrun` DDP, `no_sync` with accumulation, rank-aware data sharding, a full train-state checkpoint (model, optimizer, scheduler (bf16 needs no GradScaler), RNG states for all ranks, loader position, step/tokens), atomic writes, and auto-resume.
- Evidence: 1-GPU vs 2/3-GPU loss curves overlay (same global batch); a kill-and-resume test with loss continuing on the same trajectory; a scaling-efficiency table over PCIe.
- Stretch [O]: FSDP2 `fully_shard` + DCP checkpoint, with a resharding resume from 3→2 GPUs.

**Lab 8 — Main pretraining run and base-model evaluation [E]** · 6–10 h of your time · budget from Lab 6
- Question: *What did my budget buy, and how do I measure a base model honestly?*
- You do: launch with a pre-registered hypothesis in REPORT.md; monitor; evaluate held-out val loss, `lm_eval` zero-shot (HellaSwag, ARC-e/c, PIQA, Winogrande) vs. a random baseline and vs. SmolLM2-135M (labelled as a reference trained on ~2T tokens); generate samples at fixed seeds; export to HF format and verify that `AutoModelForCausalLM` reproduces your logits.
- While it trains, implement KV-cache generation (`infer/kv_cache.py`) with an equivalence test against no-cache decoding.

### M4 — Supervised post-training  [CME 295 L3]
**Exit criteria:** SFT model with *proven* assistant-only masking; a LoRA vs full FT comparison with clearly stated tradeoffs.

**Lab 9 — SFT: chat templates, special tokens, loss masking [E]** · 8–12 h · 3–6 GPU-h
- You implement: a ChatML template (with `{% generation %}` tags so TRL can also mask), tokenization of conversations, an assistant-only label mask, the SFT loop in your own code; then reproduce with TRL `SFTTrainer` (`assistant_only_loss=True`) as a cross-check.
- Data: `HuggingFaceTB/smol-smoltalk` (Apache-2.0).
- Evidence: a colored token/mask visualization for 5 examples, a test that every supervised token is inside an assistant turn and that `<|im_end|>` is supervised, before/after samples, and your loop vs TRL curves.

**Lab 10 — Full fine-tuning vs LoRA [E, short]** · 5–8 h · 3–6 GPU-h
- You implement: the LoRA math in a small module (A/B, scaling, merge) and test it against PEFT; then use PEFT for runs.
- Evidence: a quality/memory/time table across ranks and target modules, forgetting measured on base val loss, and a merge-equivalence test.

### M5 — Preferences  [CME 295 L3]
**Exit criteria:** an RM with held-out accuracy and calibration analysis; a DPO model evaluated beyond training loss.

**Lab 11 — Preference data, rubrics, reward modeling [E]** · 8–12 h · 3–6 GPU-h
- You do: write an annotation rubric and hand-label ~150–300 pairs of your own SFT model's outputs (**human preferences**); compare with **AI-judge preferences** (UltraFeedback ratings are GPT-4 judgments, so they are AI feedback, not human) and with **programmatic** labels; measure agreement.
- You implement: a Bradley–Terry RM (scalar head on the last non-pad token).
- Evidence: inter-source agreement, RM held-out accuracy, a length-bias check (does the RM just prefer longer answers?).

**Lab 12 — DPO [E]** · 5–8 h · 3–6 GPU-h
- You implement: the DPO loss from log-probs (policy and frozen reference), with tests on hand-computed values; then cross-check with TRL `DPOTrainer`.
- Evidence: reward margins/accuracy, KL to reference, win-rate vs SFT judged by your RM **and** a held-out judge, and length drift. The control model is used where the 135M model is too weak.

### M6 — Online RL  [CME 295 L4]
**Exit criteria:** a GRPO run with reward diagnostics and held-out generalization tests; a PPO implementation passing unit checks and a small stable run.

**Lab 13 — GRPO with verifiable rewards [E]** · 10–15 h · 6–12 GPU-h
- Task design: procedural tasks with exact checkers and controllable difficulty (2–3 digit arithmetic, list sorting, string reversal, strict output format). First, a light SFT on the task format brings pass@8 into **~10–60%**. Below that, the reward signal is almost always zero; above it, there is little left to learn.
- You implement: group sampling, group-relative advantages (with and without std scaling, per Dr. GRPO), clipped token-level loss, optional KL; then compare with TRL `GRPOTrainer`. **Know its defaults:** `loss_type="dapo"`, `beta=0.0` (no KL), `scale_rewards="group"`.
- Diagnostics: the fraction of groups with zero reward variance (uninformative prompts); reward vs. length; entropy; a held-out **harder/out-of-distribution** split (more digits, new phrasing) to detect non-generalizing gains; and a manual review of 30 high-reward samples for reward hacking (e.g. format-only exploits, checker loopholes).
- Stretch [O]: vLLM server mode (1 GPU generation + 2 GPU training), or Countdown on the control model.

**Lab 14 — PPO-based RLHF [E-advanced; first candidate to cut]** · 12–20 h · 8–16 GPU-h
- You implement: rollout → reward (your RM from Lab 11) − β·KL(ref) per token → value head → GAE → whitening → clipped policy and value losses → multiple epochs over minibatches. Follow the N+ implementation details (EOS handling, reward on the last token, value-head init).
- Evidence: unit tests for GAE and the losses on toy tensors; curves for KL, clip-fraction, value explained-variance, approx-KL and RM score; RM over-optimization (proxy RM up while held-out judge down) shown and discussed.
- Optional cross-check: `trl==1.12.0` `trl.experimental.ppo` in a separate venv.

### M7 — Release  [CME 295 L5, L7]
**Exit criteria:** someone else can reproduce headline numbers from the README; every claimed skill maps to evidence.

**Lab 15 — Final evaluation, inference demo, reproducibility audit, release [E]** · 8–12 h · 2–4 GPU-h
- Evidence: an evaluation report across all checkpoints (base → SFT → DPO → GRPO/PPO) on the same suite with seeds and confidence intervals; a KV-cache inference demo (CLI or small Gradio/Space); a clean-clone reproduction of one small run by following the README; model card + data card; public HF releases with pinned revisions; `docs/skills_evidence.md`.

## 6. RL and preference ground rules (apply in Labs 11–14)
- **Human preferences**: comparisons that people made (your own labels; hh-rlhf). **AI preferences**: a model's judgments (UltraFeedback's GPT-4 ratings, LLM-as-judge, RLAIF/Constitutional AI). **Programmatic rewards**: deterministic checkers (exact answer, format, unit tests). Reports must name which kind each signal is.
- **An uninformative reward** shows as all-zero or all-one groups, near-zero advantage variance, and flat reward from step 0. Fix the task difficulty, not the hyperparameters.
- **Reward hacking** shows as reward rising while held-out checks, a stricter checker, or human spot checks don't; length or format exploits; and KL growing fast. Always keep a *stricter* held-out verifier than the training reward.
- **Non-generalizing gains**: report in-distribution vs. OOD splits separately, and track base val loss / one capability benchmark to catch regressions.

## 7. Git and GitHub workflow (all labs)
- One **milestone** per M0–M7; one **issue** per lab (template), plus issues for bugs and experiments.
- Branch per issue: `labNN/<slug>`, e.g. `lab03/transformer-parity`. Small commits with conventional prefixes (`feat:`, `test:`, `fix:`, `docs:`, `ci:`, `exp:`) and an `AI-Assisted:` trailer (see `AI_ASSISTANCE.md`).
- A PR per lab (or sub-part), with the checklist template, CI green, self-review comments on non-obvious lines, and REPORT.md updated. Merge with merge commits so the small commits stay visible.
- Rulesets on `main`: PR required, the `ci` status check required, no force-push.
- Tags at milestones: `m0-foundations`, `m3-base-model`, …, `v1.0`.

## 8. Mentoring protocol
For each lab: concepts → interfaces/minimal scaffolding → you implement → progressive hints on request (H1 nudge → H2 approach → H3 pseudo-code → full solution only if you ask) → you share code/logs/results → review against acceptance criteria → completion summary + STATUS update. A lab is complete only with evidence. Anything I cannot verify (e.g. runs on your server) is stated as such.

## 9. Key references
CME 295 (Fall 2026) · Vaswani 2017 (1706.03762) · SmolLM2 (2502.02737) · FineWeb (2406.17557) · Chinchilla (2203.15556) · LoRA (2106.09685) · LoRA Learns Less and Forgets Less (2405.09673) · InstructGPT (2203.02155) · PPO (1707.06347) · N+ RLHF-PPO details (2403.17031) · DPO (2305.18290) · DeepSeekMath/GRPO (2402.03300) · DeepSeek-R1 (2501.12948) · Dr. GRPO (2503.20783) · DAPO (2503.14476) · Constitutional AI (2212.08073) · RLAIF (2309.00267) · Tülu 3 (2411.15124).
