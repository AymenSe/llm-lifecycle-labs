# LLM Lifecycle Labs

Training a ~135M-parameter decoder-only language model **from random initialization** on real text (FineWeb-Edu), then carrying my own checkpoints through SFT, LoRA, reward modeling, DPO, GRPO and PPO, all on 1–3 shared NVIDIA A100 40 GB GPUs.

The goal is verifiable engineering evidence: every claim below links to code, tests, configs, logs or reports in this repository. Methods follow public research and open-source tools; this repo makes no claim to reproduce any lab's proprietary recipes.

## Results
_Filled in as milestones complete (see `docs/evaluation_report.md`)._

## Roadmap and status
- Plan: [`docs/ROADMAP.md`](docs/ROADMAP.md)
- Progress: [`STATUS.md`](STATUS.md)
- Skills → evidence: [`docs/skills_evidence.md`](docs/skills_evidence.md) (built in Lab 15)
- AI assistance: [`AI_ASSISTANCE.md`](AI_ASSISTANCE.md)

## Quickstart
```bash
git clone <repo-url> && cd llm-lifecycle-labs
uv sync --locked --extra cu126   # GPU server (or --extra cpu)
uv run pytest -q
uv run python scripts/env_report.py --out runs/env.json
```

## License
Code: <MIT/Apache-2.0>. Data and model licenses: see `docs/data_card.md` and `docs/model_card.md`.
