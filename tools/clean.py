import json
import os
from pathlib import Path
from datetime import datetime, timezone

folder = Path("systems/non_click")
cutoff = datetime(2026, 2, 1, tzinfo=timezone.utc)

removed = 0
kept = 0

for f in folder.glob("*.json"):
    with open(f) as fp:
        data = json.load(fp)
    
    first_seen = data.get("evidence", {}).get("journal", {}).get("first_seen_utc")
    if first_seen is None:
        print(f"SKIP (no date): {f.name}")
        continue
    
    dt = datetime.fromisoformat(first_seen.replace("Z", "+00:00"))
    if dt < cutoff:
        os.remove(f)
        removed += 1
    else:
        kept += 1

print(f"Supprimés: {removed} | Conservés: {kept}")