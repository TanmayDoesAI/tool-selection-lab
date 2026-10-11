# The study: 100 tools, one agent

**Question.** An agent has 100 tools, and some of them look alike. Should it see every tool, have the request routed
to a category first, get a retrieved top-k shortlist, or search for tools itself? Which approach completes the most
tasks, and at what cost in tokens and latency?

This folder holds everything used to answer that: a deterministic, fictional retail backend with 100 tools, the
task sets, nine discovery strategies, the runner, the graders and the analysis. The hypotheses and every change to
the protocol were written down before the results were seen, in [`docs/preregistration.md`](docs/preregistration.md)
(hypotheses H1-H14 and dated amendments A1-A17). The results are published separately; see [`results/`](results/).

## Setup

- **World.** Northstar Retail is a fictional shop with 100 tools in 10 categories of 10 (orders, inventory, shipping,
  customers, billing, support, marketing, catalog, returns, analytics). Tools read and change a local, deep-copied
  fixture, so no call reaches a real service. The catalog has deliberate near-collisions; for example, there are three
  `set_*_address` tools in different categories. The full catalog is in [`docs/tool-catalog.json`](docs/tool-catalog.json).
- **Tasks.** `data/haiku.json` is the primary evaluation set of **329 tasks**: direct, paraphrase, confusable,
  multi-step, clarification and unsupported requests. These were written by independent Haiku agents that never saw
  the strategies or the other task sets ([`docs/haiku-authoring/`](docs/haiku-authoring/) records how). Gold outcomes
  are computed by running each reference trace, not typed by hand. `data/dev.json` (60 tasks) was used for tuning and
  `data/test.json` (40 tasks) is frozen. `data/freeze.json` holds hashes that make any edit to a task file fail validation.
- **Strategies** (the nine core arms, defined in `src/toolbench/strategies.py`):

  | Arm | What the agent sees |
  |---|---|
  | `all` | all 100 tool schemas |
  | `category` | the execution model first picks up to 2 of the 10 categories (one extra call) |
  | `category-laya` | a small local decision model (Laya, run on the CPU) picks the top 2 categories; routes are cached in `data/laya_routes.json` |
  | `semantic@10` | top 10 by averaged spaCy word vectors (cosine) |
  | `keyword@10` | top 10 by BM25 over name, description and parameter names |
  | `semantic-nim@10` | top 10 by NVIDIA `nemotron-3-embed-1b` sentence embeddings |
  | `agent-search` | only a `search_tools` tool (Nemotron embeddings, 5 results per search); found tools become callable |
  | `agent-search-kw` | the same, with BM25 search |
  | `hybrid` | the `semantic-nim@10` shortlist plus `search_tools` |

  A k-sweep (`semantic-nim@3/5/20/40`) adds points between k=10 and `all` (k=100).
- **Models.** The study ran on NVIDIA NIM models through the OpenAI-compatible API: `openai/gpt-oss-20b`,
  `nvidia/nemotron-3.5-lightning-30b-a3b` and `nvidia/nemotron-3-super-120b-a12b` were run to completion, and
  `z-ai/glm-5.3-flash` and `meta/muse-glimmer-30b` were run in part. All models use the same system prompt,
  temperature 0, up to 8 steps, a 120 s timeout, seed 1741 and one repetition.
- **Grading** (`src/toolbench/grading.py`) is based on outcomes. A task is complete only if the final fixture state,
  the facts in the reply, and the final status (done, asked for clarification, or declined) all match the gold result.
  Tool choice, argument validity and failure class are scored separately. For example, `retrieval_failure` means the
  needed tool was never shown to the agent.

## Reproduce

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). The first sync downloads the pinned spaCy model wheel.

```bash
cd study
uv sync --frozen
uv run --frozen toolbench validate        # 100 tools, frozen task hashes
uv run --frozen pytest -q                 # offline; HTTP is mocked
uv run --frozen toolbench offline         # every reference trace passes the real grader, no inference
```

To run one model through all nine arms on the 329 haiku tasks, export your NVIDIA NIM API key as `NIM_API_KEY` and run:

```bash
uv run --frozen toolbench benchmark --provider nim --model openai/gpt-oss-20b --split haiku \
  --strategies all category category-laya semantic@10 keyword@10 semantic-nim@10 agent-search agent-search-kw hybrid \
  --max-steps 8 --timeout 120 --max-requests 30000 --concurrency 4 --out runs/haiku-core-gpt-oss-20b
uv run --frozen toolbench resume runs/haiku-core-gpt-oss-20b          # continue after an interruption
uv run --frozen python scripts/make_repair.py runs/haiku-core-gpt-oss-20b   # re-run api_error jobs once (A11)
uv run --frozen toolbench resume runs/haiku-repair-gpt-oss-20b
uv run --frozen python scripts/analyze.py runs/haiku-core-gpt-oss-20b runs/haiku-repair-gpt-oss-20b --out analysis.json
```

Runs are append-only (`attempts.jsonl`, `results.jsonl`) with a `checkpoint.json` holding the plan. `resume` skips
finished jobs and refuses to continue if the protocol code has changed. `toolbench report <run>` writes a portable HTML
report with full traces. `--provider groq` (with `GROQ_API_KEY`) also works; the early smoke runs used it.

Other scripts: `scripts/coverage.py` measures offline shortlist coverage at several k; `scripts/compile_tasks.py`
turns raw authored tasks into graded tasks by executing them; `scripts/expects.py` checks the requested-fact
annotations; `docs/laya/laya_router.py` precomputes the Laya routes.

## Layout

```
src/toolbench/   world.py (fixtures) · tools.py (100 tools) · strategies.py · api.py (client, retries, budget)
                 runner.py (paired plan, checkpoint/resume) · grading.py · report.py · cli.py
data/            dev / test / haiku task sets, freeze hashes, fact annotations, cached Laya routes
docs/            design.md · preregistration.md (+ sha256) · tool-catalog.json · sources.md
                 haiku-authoring/ (task authoring brief, raw and curated tasks) · laya/ (router feasibility)
scripts/         analyze.py · make_repair.py · coverage.py · compile_tasks.py · expects.py
tests/           offline tests
results/         where the final exports go (published separately for now)
```

## Limitations

The retailer, tools and fixtures are synthetic, so the tasks are not real traffic. The categories are balanced, IDs
are stable, and the vocabulary is shared between tools and tasks, all of which may make discovery easier than in
practice. Shortlists are fixed from the first message, except in the search arms. Provider-side errors and timeouts
are recorded as `api_error`, re-run once in a repair run, and reported per model and arm rather than counted as task
failures. `docs/design.md` lists the design challenges and the grading audit.
