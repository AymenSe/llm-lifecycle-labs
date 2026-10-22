# AI Assistance Log

This project uses an AI mentor (Claude) for explanations, reviews and hints. This file records what kind of help each piece of work received, so readers can tell **code I designed, wrote and can explain** apart from **text or code that was generated for me**.

## Levels
| Level | Meaning |
|---|---|
| `none` | Written without AI input |
| `explain` | AI explained concepts or papers; I wrote all code/text |
| `review` | AI reviewed my work; I made the changes myself |
| `hint` | AI gave H1–H3 hints (nudge / approach / pseudo-code); I wrote the code |
| `scaffold` | AI provided interfaces, configs or boilerplate that I adapted (e.g. the pyproject index block, lab READMEs) |
| `generated` | AI wrote the code/text; I reviewed and tested it. Must be listed below with how it was verified |

## Commit trailer
Every commit carries one line: `AI-Assisted: <level>[, <level>]`, e.g. `AI-Assisted: hint, review`.

## Log
| Date | Lab | Artifact (file / PR) | Level | What the AI contributed | What I did & how I verified |
|---|---|---|---|---|---|
| 2026-10-21 | all | `docs/ROADMAP.md`, `labs/lab00-setup/README.md`, `STATUS.md`, this file | scaffold | Curriculum design and lab instructions | Reviewed; adapting as the project evolves |
| 2026-10-22 | 0 | `docs/decisions/0001-stack.md`, REPORT answer on `explicit` | generated | Wrote ADR text and explanation | Read and verified against my server facts (driver 560 / CUDA 12.6) |
