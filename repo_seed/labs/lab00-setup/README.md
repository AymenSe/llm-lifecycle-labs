# Lab 0 — Environment, Hardware Inspection, and Git/GitHub Workflow

**Milestone:** M0 Foundations · **Status:** essential · **Branch:** `lab00/env-setup`

## 1. Engineering question
> *Can someone else, or me in six months, rebuild my exact software environment from a fresh clone, and what can my GPUs actually deliver compared with their spec sheet?*

Every later number (throughput, loss, eval score) is only meaningful if you know the exact code revision, package versions, driver, GPU state and topology that produced it. This lab builds that foundation and the GitHub habits you'll use for the next 15 labs.

## 2. Learning objectives
By the end you can:
1. Explain the difference between the **NVIDIA driver**, the **CUDA runtime bundled in the PyTorch wheel**, and a **system CUDA toolkit (nvcc)**, and pick a PyTorch CUDA index from the driver version.
2. Build a **locked, reproducible** uv environment with separate CPU (CI) and CUDA (server) variants.
3. Inspect GPUs, interconnect topology and the state of a **shared** machine, and record it as machine-readable provenance.
4. Measure achieved matmul TFLOPS and memory bandwidth **correctly** (warm-up, synchronization, robust statistics) and explain the gap to peak.
5. Run a GitHub workflow: milestones, issues, feature branch, small commits, PR with checklist, required CI, secret/large-file protection.

## 3. Prerequisites
- SSH access to the Linux GPU server; a GitHub account; a Hugging Face account; a W&B account (the academic tier is free) or none if you prefer trackio later.
- Recommended editor setup: VS Code (or similar) on Windows with **Remote-SSH**, so the repo lives on the server. Your Windows `LLM_Training` folder keeps personal notes and course PDFs. **Do not commit CME 295 slides**: they are not yours to redistribute.

## 4. Theory (short)

**Driver vs. runtime vs. toolkit.** The kernel driver (`nvidia-smi` header, e.g. "Driver Version 5xx, CUDA Version 12.x/13.x") states the *maximum* CUDA runtime it can run. PyTorch wheels ship their own CUDA runtime, cuDNN and NCCL as pip packages, so you **don't need a system CUDA toolkit** unless you compile extensions. Rule: pick the newest PyTorch index (`cu130`, `cu126`) whose CUDA version is ≤ the driver's reported CUDA version. PyTorch 2.13 publishes stable CUDA 12.6 and 13.0 builds; both support A100 (sm_80).

**Why torch 2.13.0 and not 2.14.1?** vLLM 0.31.0, the newest vLLM that TRL 1.16 accepts, pins `torch==2.13.0`. Pinning now avoids a mid-project migration before Lab 13. Record this in an ADR.

**Lockfiles.** `pyproject.toml` states *intent* (ranges). `uv.lock` records the *resolution* (exact versions + hashes for every platform). `uv sync --locked` fails if the lock is stale; that is your reproducibility guarantee. CI should use `--locked`.

**Measuring GPU performance.** CUDA kernels launch asynchronously, so timing without `torch.cuda.synchronize()` (or CUDA events) measures launch overhead, not compute. Warm up first (cuBLAS heuristics, clocks), then report the **median** over many iterations. FLOPs of an (m×k)·(k×n) matmul = **2·m·n·k**. A100 PCIe 40 GB spec: **312 TFLOPS dense BF16/FP16**, 156 TF32, 19.5 FP32, **1,555 GB/s** HBM2, 250 W TDP. Large square matmuls typically reach a substantial fraction of peak; skinny matmuls (like your model's d=576 layers) reach much less. That is a central theme of Lab 6.

**TF32.** On Ampere, FP32 matmuls can run on tensor cores with a 10-bit mantissa (`torch.backends.cuda.matmul.allow_tf32` / `torch.set_float32_matmul_precision("high")`). It is faster with lower precision. You must know which mode your benchmarks and training use.

**Topology.** PCIe A100s can be NVLink-bridged **only in pairs** (600 GB/s). Otherwise they talk over PCIe Gen4 (~64 GB/s theoretical, less in practice). With 3 GPUs, at most one pair can be bridged, so DDP all-reduce across all 3 is bounded by PCIe. `nvidia-smi topo -m` tells you which case you have (`NV#` = NVLink; `PIX/PXB/PHB/NODE/SYS` = increasingly distant PCIe paths).

**Shared GPUs.** Use `CUDA_DEVICE_ORDER=PCI_BUS_ID` so that CUDA indices match `nvidia-smi` indices, and select free GPUs explicitly with `CUDA_VISIBLE_DEVICES`. Benchmarks taken while someone else uses a GPU are invalid; your report must detect and flag that.

**Git as evidence.** A reviewer trusts history that shows small, test-backed, explained changes: issues stating intent, PRs with self-review, CI gates. Large artifacts and secrets never enter Git; they are *referenced* (HF repo + revision, or path + sha256).

## 5. Task list

### 0.1 GitHub project scaffolding (≈45 min)
1. Create a **public** repo (suggested name `llm-lifecycle-labs`) with MIT or Apache-2.0 license and a Python `.gitignore`.
2. Create milestones **M0 Foundations … M7 Release** (names from `docs/ROADMAP.md`).
3. Labels: `lab`, `experiment`, `bug`, `infra`, `docs`.
4. Add `.github/ISSUE_TEMPLATE/lab.md` (fields: question, objectives, tasks checklist, acceptance criteria, evidence links) and `.github/pull_request_template.md` (checklist below).
5. Open issues for Lab 0, 1, 2 under their milestones. Work on Lab 0 from branch `lab00/env-setup`.
6. Add a **ruleset** on `main`: require PR, require the `ci` status check, block force-push. (You are solo, so don't require approving reviews; that would block you.)

### 0.2 Server preparation (≈45 min)
1. Configure Git identity; authenticate to GitHub (SSH key or `gh auth login`).
2. Install uv (user-level). Choose a **data disk** with ≥200 GB free and set, e.g. in `~/.bashrc`:
   `LMLAB_ROOT`, `UV_CACHE_DIR`, `HF_HOME` → all on the data disk (home-directory quotas are a classic failure).
   Rough disk needs: raw sample-10BT 28.5 GB + uint16 shards ~2 bytes/token + ~2 GB per full train-state checkpoint.
3. Record `nvidia-smi`, `nvidia-smi topo -m`, `nvidia-smi -q -d CLOCK,POWER` (power cap!) and MIG status. Decide the CUDA index from the driver line.
4. Authenticate tokens **outside the repo**: `hf auth login` (fine-grained token, write access only to your own repos) and `wandb login`. Verify with `hf auth whoami`.

### 0.3 Reproducible environment (≈1 h)
1. `uv init --package --name lmlab --python 3.12` (src layout).
2. Add dependencies **for Lab 0 only**: `torch==2.13.0`, `numpy`, `pyyaml`, `nvidia-ml-py` (NVML bindings) or parse `nvidia-smi`; dev group: `pytest`, `ruff`, `pre-commit`. Later labs add their dependencies in their own PRs.
3. Configure torch indexes with **CPU and CUDA extras declared as conflicting** (pattern from the uv PyTorch guide). Replace `cu126` with `cu130` if your driver supports CUDA 13:

```toml
[project.optional-dependencies]
cpu   = ["torch==2.13.0"]
cu126 = ["torch==2.13.0"]

[tool.uv]
conflicts = [[{ extra = "cpu" }, { extra = "cu126" }]]

[tool.uv.sources]
torch = [
  { index = "pytorch-cpu",   extra = "cpu" },
  { index = "pytorch-cu126", extra = "cu126" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```
4. `uv lock`, then on the server `uv sync --locked --extra cu126`; in CI `uv sync --locked --extra cpu`.
5. Write `docs/decisions/0001-stack.md` (ADR: context, decision, alternatives, consequences). It must cover Python 3.12, torch 2.13.0 + chosen CUDA index (with the driver evidence), uv, and the planned transformers `>=5.10.4,<5.18` constraint to keep the vLLM option open.

### 0.4 Environment & hardware report (core implementation, ≈1.5 h)
Implement `src/lmlab/utils/env.py` and `scripts/env_report.py`. This module is reused by **every** training run later, so design it carefully.

```python
# src/lmlab/utils/env.py — interfaces (you implement)
def git_state(repo_root: Path) -> dict:
    """{'sha': str, 'branch': str, 'dirty': bool, 'untracked': int}. Never raises outside a git repo; returns sha=None."""

def gpu_inventory() -> list[dict]:
    """One dict per *physical* GPU: index, name, pci_bus_id, mem_total_gib, mem_used_gib,
    utilization_pct, compute_capability, power_limit_w, n_processes. Empty list if no NVIDIA GPU."""

def torch_info() -> dict:
    """torch version, torch.version.cuda, cudnn, nccl, cuda_available, visible_device_count,
    bf16_supported, tf32_matmul_allowed, float32_matmul_precision."""

def collect_env(include_gpu: bool = True) -> dict:
    """Top-level report: schema_version, timestamp_utc, python, platform, packages (pinned subset),
    torch, gpus, topology (raw `nvidia-smi topo -m` text or None), cpu, ram_gib,
    disk {path, free_gib} for $LMLAB_ROOT, git, env vars (CUDA_VISIBLE_DEVICES, CUDA_DEVICE_ORDER),
    warnings: list[str]  # e.g. 'GPU 1 busy: 23 GiB used by 2 processes'."""
```
Constraints: JSON-serializable; **no secrets** (never dump the full environment); omit the hostname or hash it; works on a CPU-only CI runner.

`scripts/env_report.py --out <path>` writes the JSON. Commit one sanitized server report as `labs/lab00-setup/results/env_report.json`.

### 0.5 GPU micro-benchmarks (core implementation, ≈1.5 h)
Implement `src/lmlab/utils/bench.py` and `scripts/bench_gpu.py`:

```python
def matmul_flops(m: int, n: int, k: int) -> int: ...
def time_cuda(fn, warmup: int = 10, iters: int = 50) -> dict:
    """Uses torch.cuda.Event or synchronize; returns median_ms, p10_ms, p90_ms, all in ms."""
def bench_matmul(m, n, k, dtype, device="cuda") -> dict:   # + achieved TFLOPS and % of spec peak
def bench_copy_bandwidth(nbytes: int, device="cuda") -> dict:  # read+write bytes / time → GB/s
```
Experiments (single GPU, chosen free via `CUDA_VISIBLE_DEVICES`):
1. Square matmuls n ∈ {1024, 2048, 4096, 8192} × dtype ∈ {bf16, fp16, fp32 with TF32 off, fp32 with TF32 on}.
2. **Your model's real shapes**, with B·T = 16384 tokens: (16384×576)·(576×1536) [MLP up], (16384×1536)·(1536×576) [MLP down], (16384×576)·(576×576) [attn proj], (16384×576)·(576×49152) [LM head], in bf16.
3. Device copy bandwidth with a ≥1 GiB tensor.
4. Save `results/bench_gpu.json` and a plot `results/matmul_tflops.png` (TFLOPS vs. size, one line per dtype, a dashed line at the spec peak).

### 0.6 CI, hooks, hygiene (≈1 h)
1. `.github/workflows/ci.yml`: on `pull_request` and push to `main`; job named `ci`; steps: checkout → `astral-sh/setup-uv` (use the current major version from its README) → `uv sync --locked --extra cpu` → `uv run ruff check .` → `uv run ruff format --check .` → `uv run pytest -q`.
2. `.pre-commit-config.yaml` with `pre-commit-hooks` (`check-added-large-files` with `--maxkb=1000`, `detect-private-key`, `check-yaml`, `end-of-file-fixer`) and `ruff-pre-commit`. Pin revisions with `pre-commit autoupdate`; don't guess versions.
3. `.gitignore` additionally covers: `.env`, `artifacts/`, `runs/`, `data/`, `checkpoints/`, `wandb/`, `*.pt`, `*.safetensors`, `*.bin`, `*.parquet`, `*.npy`, `*.arrow`.
4. Tests in `tests/` (CPU-only; GPU tests use `@pytest.mark.gpu` and are skipped in CI):
   - `matmul_flops` correctness;
   - `collect_env(include_gpu=False)` returns all required keys and round-trips through `json.dumps`;
   - `git_state` detects a dirty working tree in a temporary repo;
   - `time_cuda`-equivalent CPU path returns finite positive values (or design `time_fn` to be device-aware);
   - a **"no secrets"** test: the report contains no env-var values matching `*TOKEN*`, `*KEY*`, `*SECRET*`.

### 0.7 Documentation (≈45 min)
Copy in `README.md` (skeleton), `STATUS.md`, `AI_ASSISTANCE.md`, `docs/ROADMAP.md` and this file from the mentor seed. Write `labs/lab00-setup/REPORT.md` (structure in §11).

## 6. You implement vs. libraries
| You implement | Libraries may handle |
|---|---|
| `env.py` report design and logic, `bench.py` timing and FLOP accounting, tests, plot script, ADR, REPORT | uv (resolution/lock), NVML or `nvidia-smi` (raw data), matplotlib, GitHub Actions, pre-commit hooks |
| The pyproject index/extras config (given above; you must be able to explain each line) | |

## 7. Expected files after Lab 0
```
pyproject.toml  uv.lock  .python-version  .gitignore  .pre-commit-config.yaml
.github/workflows/ci.yml  .github/ISSUE_TEMPLATE/lab.md  .github/pull_request_template.md
src/lmlab/__init__.py  src/lmlab/utils/{__init__,env,bench}.py
scripts/{env_report,bench_gpu,plot_bench}.py
tests/{test_env,test_bench}.py
docs/ROADMAP.md  docs/decisions/0001-stack.md
labs/lab00-setup/{README.md,REPORT.md,results/env_report.json,results/bench_gpu.json,results/matmul_tflops.png}
README.md  STATUS.md  AI_ASSISTANCE.md
```

## 8. Debug configuration vs. larger experiment
- **Debug (CPU, seconds):** `bench_matmul` with n=256 fp32 on CPU through the same code path; `collect_env(include_gpu=False)`. This is what CI runs.
- **Main (1 free GPU, <15 min):** experiments 1–3 from §0.5.
- **Optional larger (2–3 GPUs, when free):** the stretch NCCL all-reduce benchmark (§14).

## 9. Effort and compute
- Your time: **5–8 h** (likely 1.5 weeks at your pace). The largest uncertainty is server-side surprises (driver, quotas, proxy, GPU contention).
- Compute: <0.5 GPU-hour. Disk: negligible for this lab; verify ≥200 GB is available for Labs 1–8.

## 10. Correctness checks and acceptance criteria
Lab 0 is complete when **all** of these are observable:

| # | Criterion | Evidence |
|---|---|---|
| A1 | A fresh clone on the server reproduces the environment: `uv sync --locked --extra <cuXXX>` succeeds without modifying `uv.lock` | Command + output snippet in REPORT |
| A2 | `torch.version.cuda` matches the chosen index; `cuda_available=True`; compute capability `[8, 0]`; `bf16_supported=True` | `results/env_report.json` |
| A3 | The ADR justifies the CUDA index with the driver's reported CUDA version | `docs/decisions/0001-stack.md` |
| A4 | The env report includes git sha + dirty flag, topology text, per-GPU memory in use, and warnings for busy GPUs; it contains no secrets or hostname | JSON + passing "no secrets" test |
| A5 | Benchmarks report median and p10/p90 over ≥50 timed iterations after warm-up, with explicit synchronization | code review + JSON |
| A6 | Large bf16 matmul reaches a plausible fraction of the 312 TFLOPS spec (≥50%). If it is lower, the REPORT gives a diagnosed reason (power cap, clocks, contention, wrong dtype) | JSON + REPORT |
| A7 | TF32-on vs TF32-off FP32 results differ, and the REPORT explains why | plot + REPORT |
| A8 | The model-shape benchmark is reported, and the REPORT explains why the d=576 matmuls achieve lower TFLOPS than n=8192 | JSON + REPORT |
| A9 | Copy bandwidth is reported as a % of 1,555 GB/s, with the byte-counting convention stated | JSON |
| A10 | CI runs on the PR; the `ci` check is required on `main`. **A deliberately failing test commit shows CI red, then green after the fix** (both visible in the PR history) | PR link |
| A11 | Pre-commit blocks (a) a fake file containing a `-----BEGIN RSA PRIVATE KEY-----` header and (b) a >1 MB file. The blocked attempts are described in REPORT, not committed | REPORT |
| A12 | The PR has ≥4 focused commits with conventional prefixes, uses the template checklist, and links the issue (`Closes #N`) | PR |
| A13 | Milestones M0–M7 and issues for Labs 0–2 exist | GitHub |
| A14 | REPORT answers the conceptual questions (§15) in your own words | REPORT |

I cannot see your server or GitHub directly. I verify by reviewing what you paste or link: JSON, PR URL, CI logs, REPORT. Anything I can't inspect will be marked "unverified" in the review.

## 11. Metrics, plots, logs to collect
- `env_report.json`, raw `nvidia-smi topo -m` output (inside the JSON).
- `bench_gpu.json`: per shape/dtype, median/p10/p90 ms, TFLOPS, % of peak; bandwidth GB/s.
- `matmul_tflops.png`.
- REPORT.md sections: **Goal & hypothesis** (write *before* running: "I expect bf16 8192² to reach ~X% of peak because…"), **Setup** (commit sha, GPU id, other users present?), **Results**, **Failures & fixes**, **Interpretation**, **Limitations**, **Answers to conceptual questions**, **AI assistance**.

## 12. Common failure modes
- **`torch.cuda.is_available()` is False**: the CPU wheel got installed (check `torch.version.cuda is None`), the wrong extra was used, or the driver is too old for cu130 ("CUDA driver version is insufficient").
- **torch resolved from PyPI instead of the PyTorch index**: missing `explicit = true` or the `[tool.uv.sources]` mapping.
- **Absurd TFLOPS (e.g. >312)**: no synchronization, or FLOPs counted as m·n·k instead of 2·m·n·k.
- **Low TFLOPS**: another job on the GPU (`nvidia-smi` processes), a reduced power limit, thermal throttling, TF32 silently on/off, a too-small matrix, or the first-call cuBLAS warm-up included in the timing.
- **GPU indices don't match `nvidia-smi`**: set `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- **Disk/quota errors**: uv/HF caches defaulting to `$HOME`.
- **CI passes locally but fails on GitHub**: a lockfile not committed, `--locked` with a stale lock, tests that implicitly require a GPU.
- **Pre-commit not triggering**: `pre-commit install` was never run in the clone.

## 13. Deliverables to commit
Everything in §7, plus a merged PR closing the Lab 0 issue and the tag `m0-foundations` after review.

**GitHub issue** — *"Lab 0: Reproducible environment, hardware report, GPU benchmarks, CI"* (milestone M0, label `lab`, `infra`), with the A1–A14 checklist pasted in.

**Branch** — `lab00/env-setup`.

**Suggested commit sequence** — `chore: init uv package and lockfile` → `ci: add CPU CI workflow and pre-commit` → `feat(env): environment and GPU inventory report` → `test(env): schema, dirty-tree and no-secrets tests` → `feat(bench): matmul and bandwidth benchmarks` → `exp(lab00): A100 benchmark results and plot` → `docs: ADR 0001, lab 0 report, status`.

**PR checklist (template)**
- [ ] Linked issue and milestone
- [ ] Scope is limited to the issue; no unrelated changes
- [ ] Tests added/updated; CI green
- [ ] No data, checkpoints, secrets or files >1 MB
- [ ] Results reproducible: commands, commit sha and config recorded in REPORT
- [ ] REPORT.md updated (hypothesis, results, failures, limitations)
- [ ] AI assistance logged in `AI_ASSISTANCE.md` and commit trailers
- [ ] STATUS.md updated
- [ ] Self-review: I can explain every line I'm merging

## 14. Stretch task (optional)
NCCL all-reduce benchmark with `torchrun --nproc_per_node={2,3}`: all-reduce bf16 tensors from 1 MB to 1 GB; report algorithm bandwidth and **bus bandwidth** (busbw = algbw × 2(n−1)/n). Compare the pair that `topo -m` shows as closest with the full trio. Then estimate the per-step cost of all-reducing 134.5M bf16 gradients (≈269 MB) and keep that number for Lab 7.

## 15. Conceptual questions (answer in REPORT)
1. Your driver reports "CUDA Version 12.8". Can you use the `cu130` wheel? The `cu126` wheel? Do you need `nvcc`? Why?
2. What does `uv sync --locked` guarantee that `pip install -r requirements.txt` with `>=` ranges does not?
3. Your bf16 8192² matmul reaches X TFLOPS but the 576×1536 model matmul reaches far less. Give two reasons, using the terms arithmetic intensity and tile/wave quantization.
4. Your `topo -m` output shows the relationship between GPU0 and GPU2. What does that imply for 3-GPU DDP vs. 2-GPU DDP on the closest pair?
5. Why must checkpoints and datasets stay out of Git, and what exactly will you store in Git so that they remain referenceable and verifiable?
6. TF32 keeps FP32's exponent range but has a 10-bit mantissa. Why is that usually harmless for training matmuls, and where could it matter?
7. Why is timing the first iteration misleading on a GPU?

## 16. How to request help
Ask for hints by level: **H1** (nudge), **H2** (approach), **H3** (pseudo-code). I'll give a full solution only if you explicitly ask. When done, share the PR link (or paste the diff), `env_report.json`, `bench_gpu.json`, the plot, and REPORT.md for review.
