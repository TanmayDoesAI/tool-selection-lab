# The study

A fictional retailer, **Northstar**, with 100 tools in 10 categories, many of them deliberately similar
(e.g. three `set_*_address` tools). An agent gets 329 requests and one of nine ways to discover tools. A task counts
as done only if the final data and the reply are right.

| Strategy | What the agent sees |
|---|---|
| `all` | all 100 tools |
| `category` | the model picks 2 of 10 categories first |
| `category-laya` | a small local model picks the categories |
| `semantic@10` | top 10 by word vectors (spaCy) |
| `keyword@10` | top 10 by BM25 |
| `semantic-nim@10` | top 10 by NVIDIA Nemotron embeddings |
| `agent-search` | a `search_tools` tool only (embeddings) |
| `agent-search-kw` | the same with BM25 |
| `hybrid` | the `semantic-nim@10` shortlist plus `search_tools` |

## Run it

```bash
pip install -r requirements.txt && pip install --no-deps -e .    # or: uv sync
toolbench validate && toolbench offline                          # checks the tools, tasks and grader, no API calls

export NIM_API_KEY=...
toolbench benchmark --provider nim --model openai/gpt-oss-20b --split haiku --max-steps 8 --timeout 120 \
  --strategies all category category-laya semantic@10 keyword@10 semantic-nim@10 agent-search agent-search-kw hybrid \
  --max-requests 30000 --out runs/gpt-oss-20b
python scripts/analyze.py runs/gpt-oss-20b --out analysis.json
```

Runs resume where they stopped (`toolbench resume runs/gpt-oss-20b`).

## What's here

```
src/toolbench/   the world, the 100 tools, strategies, runner and grader
data/            tools.json · haiku.json (the 329 tasks) · dev/test sets · task-authoring.json (how tasks were written)
scripts/         analyze.py (paired comparison of strategies) · export_kit.py (this test in toolpick format)
tests/           offline tests (pytest)
```
