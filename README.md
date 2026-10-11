Every 5 minutes, push gzip snapshots of live benchmark runs to the private repo's `run-backups` branch.

Snapshots cut each JSONL at its last complete line, so a restore always loads. Each cycle builds a fresh
single-commit repository and force-pushes it, so neither this disk nor GitHub accumulates old snapshots.
Restore: git clone -b run-backups <repo> r && gunzip r/*/*.gz && copy the run dirs into tool-selection-lab/runs/
Stop: pkill -f tool-selection-lab-runs/backup.py
