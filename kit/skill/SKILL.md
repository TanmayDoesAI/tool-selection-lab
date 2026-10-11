---
name: toolpick-cases
description: Turn a tool catalog in any format (OpenAI or Anthropic tool JSON, MCP tools/list output, OpenAPI, Python functions, or plain prose) into toolpick's catalog.json, and write 20-30 graded test cases in cases.json, so the user can measure which tool-discovery strategy (all tools, category routing, keyword or semantic top-k, agent search, hybrid) works best for their agent (six default strategies plus optional keyword-search, spaCy and local Laya-router variants). Use when someone wants to benchmark or choose how their agent finds tools, or asks to prepare inputs for toolpick.
---

# Build toolpick inputs: catalog.json and cases.json

You are preparing two files for the toolpick kit (the `kit/` folder of this repository). The formats are defined by
`kit/schemas/catalog.schema.json` and `kit/schemas/cases.schema.json`; read both before writing anything.
Finish only when `toolpick validate catalog.json cases.json` prints `OK` with no warnings you have not explained.

## 1. Gather what the user has

Ask for, or locate in the workspace:
- **The tools.** Whatever exists: OpenAI `tools` JSON, Anthropic `tools` JSON, an MCP server's `tools/list` result,
  an OpenAPI spec, Python functions with docstrings, or a prose list.
- **The system prompt** their agent normally runs with (optional; goes in `instructions`).
- **Real or typical requests** their users send, and any known failure stories (optional, but they make the best cases).
- **How many cases** they want: 20-30. Default to 24 if they do not say.

Do not invent tools. If a tool's purpose or parameters are unclear, ask, or record your assumption in the final summary.

## 2. Write catalog.json

```json
{"name": "...", "instructions": "...optional system prompt...",
 "categories": {"billing": "Invoices, payments and refunds."},
 "tools": [{"name": "refund_payment", "description": "...", "parameters": {"type": "object", "properties": {...}, "required": [...]}, "category": "billing"}]}
```

- **Structured JSON** (OpenAI, Anthropic, MCP): run `uv run toolpick convert tools.json -o catalog.json`, then add
  categories and instructions by hand.
- **OpenAPI:** one tool per operation. `name` = operationId (or `method_path` in snake case), `description` = summary
  plus the important part of the description, `parameters` = an object schema combining path, query and body fields,
  with `required` copied over.
- **Python functions:** name = function name; description = first docstring paragraph; parameters from the type
  hints (`str` -> string, `int` -> integer, `float` -> number, `bool` -> boolean, `list[X]` -> array, `Literal` ->
  enum); parameters without defaults are required.
- **Prose:** write the schema the prose implies, and list every assumption for the user.

Rules:
- Keep the user's names and descriptions **verbatim** (only replace characters outside `A-Za-z0-9_-` in names).
  The point is to test their real catalog, so do not improve descriptions unless they ask.
- Categories: reuse the user's grouping (MCP server, API tag, module). If there is none, propose 4-10 categories
  of roughly similar size, give each a one-line description, and confirm with the user. Every tool gets one category.

## 3. Write cases.json

```json
{"mocks": {"find_order_invoices": [{"invoice_id": "I1"}]},
 "cases": [{"id": "damaged-refund", "type": "multi-step", "category": "billing",
            "prompt": "Order O100 arrived damaged. Refund $12 of what the customer paid for it.",
            "expect": {"tools": ["find_order_invoices", "refund_payment"], "ordered": true,
                       "args": {"refund_payment": {"invoice_id": "I1", "amount": 12}},
                       "allowed": ["get_order"], "forbidden": ["issue_store_credit"]}}]}
```

Field meanings:
- `type`: `single` (exactly one expected tool), `multi-step` (two or more), `unsupported` (no tool can do it; the agent
  must not call anything), `ambiguous` (a key detail is missing; the agent should ask, and may only call `allowed` lookups).
- `expect.tools`: tools that must be called. `ordered: true` when order matters (a lookup feeds the next call).
- `expect.allowed`: harmless extra calls (lookups) that should not count against the agent.
- `expect.forbidden`: near-miss tools that would be a real mistake.
- `expect.args`: `{tool: {argument: value}}` checks; `"*"` means present with any value. Strings compare case-insensitively.
- `mocks`: what tool calls return (default `{"ok": true}`). Give realistic stub results for every lookup in a multi-step
  case, so the agent can find the ID it needs for the next call (as `find_order_invoices` -> `I1` above). Use
  `{"$sequence": [...]}` for different results on successive calls, and a case-level `mocks` to override for one case.

Choose a mix close to this (for 24 cases): 12-14 single, 5-6 multi-step, 2-3 unsupported, 2-3 ambiguous.

Write good cases:
1. **Cover every category.** At least one case expects a tool from each category; spread the rest by how much the
   category is used in practice.
2. **Near-miss distractors.** Find tools that look alike (same verb or noun across categories, e.g. order vs parcel vs
   customer address; refund vs store credit). Write cases whose correct answer is one of them and put the others in
   `forbidden`. These are where discovery strategies differ most.
3. **Do not leak tool names.** Write as a user would talk: "send the money back to their card", not "call
   refund_payment". Avoid copying distinctive words from the tool description into the prompt; paraphrase. The
   validator warns when a prompt contains a tool name.
4. **Multi-step and cross-category.** Include requests that need two or more tools from different categories, as these
   are what a fixed shortlist or a single category route misses.
5. **Unsupported requests** must be plausible for this agent and close to a real tool, but not doable (e.g. "email the
   customer" when only notes exist). Never make them absurd.
6. **Ambiguous requests** must lack something the agent cannot look up (which customer, which amount).
7. **One right answer.** If two tools are both acceptable, put one in `tools` and the other in `allowed`, or rewrite the
   prompt so that only one is right. Keep `args` checks to values stated in the prompt or present in a mock.
8. Use the user's real requests when they have them; strip personal data.

## 4. Validate and hand over

```bash
cd kit
uv run toolpick validate path/to/catalog.json path/to/cases.json
```

Fix every error. For each remaining warning, either fix it or tell the user why it is fine. Then give the user:
- the case mix by type and category, and the near-miss pairs you targeted;
- every assumption you made about tools or categories;
- the next command, for example
  `uv run toolpick run --provider openai --catalog catalog.json --cases cases.json`
  (or `--provider anthropic`, `--provider nim`, `--provider openai-compatible --base-url ...`, or `--agent file.py:Class`
  for their own harness), and `--dry-run` to check the plumbing offline first.
- which strategies to run. The six defaults suit most users. Suggest the optional ones only when they fit:
  `agent-search-kw` / `hybrid-kw` (no embedding API available, or to compare keyword vs embedding search),
  `category-laya` (they want a free local router instead of an LLM routing call; needs categories and
  `pip install 'toolpick[laya]'`), `spacy@10` (a cheap local baseline; `pip install 'toolpick[spacy]'`). Example:
  `--strategies all hybrid keyword@10 category-laya`.
