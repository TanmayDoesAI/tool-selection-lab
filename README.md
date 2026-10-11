# Tool Selection Lab

**Your agent has many tools. How should it find the right one?** Show it every tool, route by category, retrieve a
top-k shortlist, or let it search?

| Folder | What it is |
|---|---|
| [`kit/`](kit/) | **toolpick**: run that comparison on *your* agent, tools and prompts. Start here. |
| [`study/`](study/) | Our benchmark: 100 tools, 329 tasks, 9 strategies, run on NVIDIA NIM models. |

## Test your agent in 3 steps

```bash
cd kit && pip install -r requirements.txt && pip install --no-deps -e .

# 1. Turn your tools into catalog.json + cases.json: point your coding agent at kit/skill/SKILL.md
# 2. Check them
toolpick validate catalog.json cases.json
# 3. Run every strategy against your model, then read report.md
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY / NIM_API_KEY with --provider anthropic / nim
toolpick run --provider openai --catalog catalog.json --cases cases.json
```

Want to try our test first? Pass `kit/examples/northstar-full/catalog.json` and `cases.json` (100 tools, 329 cases).

## Licence

MIT
