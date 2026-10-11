# Tool Selection Lab

**When an agent has many tools, how should it find the right one?** It could see every tool, route the request to a
category first, get a retrieved top-k shortlist, or search the catalog itself. This repository contains a
pre-registered study of that question on a 100-tool catalog, and a kit you can use to answer it for your own agent.

## What we found

The study compares nine discovery strategies on 329 independently written tasks, using models served through NVIDIA
NIM. The results are published separately; see [`study/results/`](study/results/). The final numbers will be added
once they are released.

## The two folders

| Folder | What it is | Start here |
|---|---|---|
| [`study/`](study/) | The experiment as it was run: a fictional retail backend with 100 tools, the task sets, nine strategies (all, category, category-laya, semantic@10, keyword@10, semantic-nim@10, agent-search, agent-search-kw, hybrid), the runner and graders, the pre-registration with its dated amendments, and the analysis scripts. | [`study/README.md`](study/README.md) |
| [`kit/`](kit/) | **toolpick**, a reusable kit to answer the same question for your own agent. Bring your tools in any format, write 20-30 test prompts with the help of the bundled skill, and get a report comparing six strategies on your model (OpenAI, Anthropic, NVIDIA NIM, any OpenAI-compatible server, or your own harness). | [`kit/README.md`](kit/README.md) |

## Reproduce the study

```bash
cd study
uv sync --frozen
uv run --frozen toolbench validate && uv run --frozen pytest -q && uv run --frozen toolbench offline
export NIM_API_KEY=...        # your NVIDIA NIM API key
uv run --frozen toolbench benchmark --provider nim --model openai/gpt-oss-20b --split haiku \
  --strategies all category category-laya semantic@10 keyword@10 semantic-nim@10 agent-search agent-search-kw hybrid \
  --max-steps 8 --timeout 120 --max-requests 30000 --out runs/haiku-core-gpt-oss-20b
uv run --frozen python scripts/analyze.py runs/haiku-core-gpt-oss-20b --out analysis.json
```

Without uv: `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt -r requirements-dev.txt && pip install --no-deps -e .`, then run the same commands without `uv run --frozen`.

The protocol, including what was decided before any results were seen, is in
[`study/docs/preregistration.md`](study/docs/preregistration.md). [`study/README.md`](study/README.md) covers the
details: the task sets, grading, repair runs and limitations.

## Test your own agent

```bash
cd kit && uv sync
# 1. Have your coding agent follow kit/skill/SKILL.md to write catalog.json and cases.json from your tools
#    (OpenAI or Anthropic tool JSON, MCP tools/list, OpenAPI, Python functions, or prose).
uv run toolpick validate catalog.json cases.json
uv run toolpick run --dry-run --catalog catalog.json --cases cases.json --out runs/dry   # offline plumbing check
export OPENAI_API_KEY=...     # or ANTHROPIC_API_KEY with --provider anthropic, or NIM_API_KEY with --provider nim
uv run toolpick run --provider openai --catalog catalog.json --cases cases.json
```

Without uv: `cd kit && python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && pip install --no-deps -e .`, then call `toolpick ...` directly.

The run writes `report.md`, `results.json` and `results.csv`, covering success rate, selection precision and recall,
tokens, latency and paired sign tests for each strategy. See [`kit/README.md`](kit/README.md) for the formats,
providers, custom harness adapters, and how to read the results.

## Sources

The external sources checked for models, APIs and the embedding and routing models are listed in
[`study/docs/sources.md`](study/docs/sources.md).

## Licence

MIT; see [`LICENSE`](LICENSE).
