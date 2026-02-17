import json
from pathlib import Path

nc_planets = 0
nc_giants = 0
nc_giants_with_orbit = 0

for p in Path("systems/non_click").glob("*.json"):
    with open(p) as f:
        d = json.load(f)
    planets = d.get("bodies", {}).get("planets", [])
    for pl in planets:
        nc_planets += 1
        pc = (pl.get("PlanetClass") or "").lower()
        if "giant" in pc:
            nc_giants += 1
            if pl.get("SemiMajorAxis"):
                nc_giants_with_orbit += 1

print(f"Planètes non_click: {nc_planets}")
print(f"Géantes non_click: {nc_giants}")
print(f"Géantes avec SemiMajorAxis: {nc_giants_with_orbit}")