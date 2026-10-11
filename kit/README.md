# toolpick

Find out which tool-discovery strategy works best for **your** agent, on **your** tools and prompts.

```bash
pip install -r requirements.txt && pip install --no-deps -e .     # or: uv sync
```

## 1. Describe your tools and write test cases

Point your coding agent (Claude Code, Codex, Cursor, ...) at [`skill/SKILL.md`](skill/SKILL.md) and paste your tools in
any format: OpenAI or Anthropic tool JSON, MCP `tools/list`, OpenAPI, Python functions, or plain text. It writes
`catalog.json` and 20-30 test cases in `cases.json`. If you already have OpenAI, Anthropic or MCP tool JSON, run
`toolpick convert my-tools.json -o catalog.json`.

```bash
toolpick validate catalog.json cases.json
```

## 2. Run

```bash
export OPENAI_API_KEY=...
toolpick run --provider openai --model gpt-4.1-mini --catalog catalog.json --cases cases.json
```

| `--provider` | Key | Example model |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | `gpt-4.1-mini` |
| `anthropic` | `ANTHROPIC_API_KEY` | `claude-haiku-5-5` |
| `nim` | `NIM_API_KEY` | `openai/gpt-oss-20b` |
| `openai-compatible` | `TOOLPICK_API_KEY` | anything behind `--base-url` (vLLM, Ollama, Groq ...) |

**Embeddings** (for `semantic@k` and search) use OpenAI `text-embedding-3-small` whenever `OPENAI_API_KEY` is set, or
NIM's when you run on NIM. Change it with `--embed-model text-embedding-3-large` or `--embed-provider`.

**Your own harness:** write a class with one method, `step(messages, tools) -> Turn`, and run
`toolpick run --agent my_agent.py:MyAgent ...`. See [`examples/custom_agent.py`](examples/custom_agent.py).

Add `--dry-run` to check everything offline first. Runs retry on rate limits and resume if you rerun the same command.

## 3. Read the report

`report.md` (also `results.json`, `results.csv`) ranks the strategies by success rate, with tokens, latency and a
paired significance test for each pair:

| Strategy | What the agent sees |
|---|---|
| `all` | every tool |
| `category` | the model picks categories first, then sees their tools |
| `keyword@k` | top-k tools by BM25 |
| `semantic@k` | top-k tools by embeddings |
| `agent-search` | a `search_tools` tool it can call |
| `hybrid` | a top-k shortlist plus `search_tools` |

**Optional strategies** (run only when you name them, e.g. `--strategies all hybrid category-laya`):

| Strategy | What it adds | Needs |
|---|---|---|
| `agent-search-kw` | agent-search that always searches with BM25 | nothing |
| `hybrid-kw` | hybrid with a BM25 shortlist and BM25 search | nothing |
| `spacy@k` | top-k by mean spaCy word vectors, fully local | `pip install 'toolpick[spacy]'` and `python -m spacy download en_core_web_md` |
| `category-laya` | a small local model ([Laya](https://huggingface.co/convaiinnovations/laya-multilingual)) picks the categories on CPU, no routing call | `pip install 'toolpick[laya]'` (PyTorch, ~1.3 GB) |

With 20-30 cases, a gap of 1-2 cases is noise. Among strategies that are not significantly behind the best, pick the
cheapest. A `not_exposed` failure means discovery hid the tool; `missing_tool` or `forbidden_tool` means the model saw
it and chose wrong, so improve the descriptions.

## Try it on our test

`examples/northstar-full/` is our full benchmark: 100 tools and 329 cases.

```bash
toolpick run --provider anthropic --catalog examples/northstar-full/catalog.json --cases examples/northstar-full/cases.json
```

With the optional strategies, the kit covers all nine strategies of our study: `semantic@10` with NIM embeddings is
the study's `semantic-nim@10`, and `spacy@10` is its `semantic@10`. It is still not a replay of the study: the kit
grades tool **selection** (right tools and arguments), while the study grades **outcomes** (final data and the facts in
the reply). Use [`../study`](../study) to reproduce our numbers exactly.

The formats are defined in [`schemas/`](schemas/). Run the tests with `pytest`.
