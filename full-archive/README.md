# Full archive (2026-10-12)

`runs-analysis-20261012.tar.gz` holds everything that was in the local working copy before it was deleted from the Pi:
`runs/` (every run directory: core, repair, smoke, check and aborted runs, logs, reports, the NIM query-embedding
cache), `analysis/` and `article-analysis/`.

Restore: `tar -xzf full-archive/runs-analysis-20261012.tar.gz -C <tool-selection-lab checkout>`
Check: `sha256sum -c full-archive/SHA256SUMS`
