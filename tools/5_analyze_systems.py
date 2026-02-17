"""
analyze_systems_v2.py
---------------------
Analyse comparative click vs non_click en exploitant les données enrichies
de systems/non_click_v2/ (schema_version 2).

Usage :
    python analyze_systems_v2.py
"""

import json
import csv
import numpy as np
from pathlib import Path
from collections import Counter

# ── Helpers ───────────────────────────────────────────────────────────────────

GAS_GIANT_CLASSES = {
    "gas giant with water based life", "gas giant with ammonia based life",
    "sudarsky class i gas giant", "sudarsky class ii gas giant",
    "sudarsky class iii gas giant", "sudarsky class iv gas giant",
    "sudarsky class v gas giant", "helium rich gas giant",
    "helium gas giant", "water giant",
}

def is_gas_giant(planet_class):
    pc = (planet_class or "").lower()
    return pc in GAS_GIANT_CLASSES or "giant" in pc

def primary_star_from_list(stars_list):
    for s in stars_list:
        if s.get("DistanceFromArrivalLS", 999) == 0.0:
            return s
    return stars_list[0] if stars_list else {}

# ── Parse metadata (click) ───────────────────────────────────────────────────

def parse_click(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    sp           = d["context"]["system"]["star_pos"]
    sm           = d.get("system_summary", {})
    bodies       = d.get("bodies", {})
    stars_list   = bodies.get("stars", [])
    planets_list = bodies.get("planets", [])
    star         = primary_star_from_list(stars_list)

    has_gas_giant    = any(is_gas_giant(p.get("PlanetClass")) for p in planets_list)
    has_ringed       = any(p.get("Rings") for p in planets_list + stars_list)
    n_terraformables = sum(1 for p in planets_list
                           if p.get("TerraformState") not in (None, "", "None"))
    n_landables      = sum(1 for p in planets_list if p.get("Landable"))

    return {
        "name":    d["context"]["system"]["name"],
        "label":   "click",
        # spatial
        "x": sp[0], "y": sp[1], "z": sp[2],
        # système
        "body_count":       sm.get("body_count"),
        "stars_count":      sm.get("stars_count"),
        "planets_count":    sm.get("planets_count"),
        "landables":        n_landables if planets_list else sm.get("landables_count"),
        "has_gas_giant":    has_gas_giant,
        "has_ringed_body":  has_ringed,
        "n_terraformables": n_terraformables,
        # étoile principale
        "primary_star_type": star.get("StarType"),
        "primary_subclass":  star.get("Subclass"),
        "stellar_mass":      star.get("StellarMass"),
        "surface_temp":      star.get("SurfaceTemperature"),
        "luminosity":        star.get("Luminosity"),
        "star_age_my":       star.get("Age_MY"),
        "has_belt":          bool(star.get("Rings")),
        "n_stars_in_system": len(stars_list) or sm.get("stars_count"),
        "n_bio_signals":     None,
        "n_geo_signals":     None,
    }

# ── Parse non_click v2 ───────────────────────────────────────────────────────

def parse_non_click_v2(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    sp    = d["system"]["star_pos"]
    feat  = d.get("features", {})
    b     = feat.get("bodies", {})
    s     = feat.get("stars", {})
    sig   = feat.get("signals", {})
    pstar = d.get("primary_star") or {}

    return {
        "name":  d["system"]["name"],
        "label": "non_click",
        # spatial
        "x": sp[0], "y": sp[1], "z": sp[2],
        # système
        "body_count":       b.get("fss_bodycount"),
        "stars_count":      s.get("n_stars"),
        "planets_count":    b.get("n_planets_scanned"),
        "landables":        b.get("n_landables"),
        "has_gas_giant":    b.get("has_gas_giant", False),
        "has_ringed_body":  b.get("has_ringed_body", False),
        "n_terraformables": b.get("n_terraformables"),
        # étoile principale (complète grâce à v2)
        "primary_star_type": pstar.get("StarType"),
        "primary_subclass":  pstar.get("Subclass"),
        "stellar_mass":      pstar.get("StellarMass"),
        "surface_temp":      pstar.get("SurfaceTemperature"),
        "luminosity":        pstar.get("Luminosity"),
        "star_age_my":       pstar.get("Age_MY"),
        "has_belt":          bool(pstar.get("Rings")),
        "n_stars_in_system": s.get("n_stars"),
        "n_bio_signals":     sig.get("n_bio_signals"),
        "n_geo_signals":     sig.get("n_geo_signals"),
    }

# ── Chargement ───────────────────────────────────────────────────────────────

records = []
errors  = []

for p in Path("metadata").glob("*.json"):
    try:
        records.append(parse_click(p))
    except Exception as e:
        errors.append(f"click {p.name}: {e}")

for p in Path("systems/non_click").glob("*.json"):
    try:
        records.append(parse_non_click_v2(p))
    except Exception as e:
        errors.append(f"non_click {p.name}: {e}")

clicks     = [r for r in records if r["label"] == "click"]
non_clicks = [r for r in records if r["label"] == "non_click"]

print(f"=== Dataset: {len(clicks)} click  |  {len(non_clicks)} non_click ===")
if errors:
    print(f"  {len(errors)} erreurs :")
    for e in errors[:5]:
        print(f"    {e}")
print()

# ── Fonctions d'affichage ─────────────────────────────────────────────────────

def stats(values, label="", fmt=".2f"):
    v = [x for x in values if x is not None]
    if not v:
        return f"  {label}: n=0 (aucune donnée)"
    return (f"  {label} (n={len(v)}): "
            f"mean={np.mean(v):{fmt}}  median={np.median(v):{fmt}}  "
            f"std={np.std(v):{fmt}}  [{min(v):{fmt}} – {max(v):{fmt}}]")

def freq(values, label="", top=8):
    v = [x for x in values if x is not None]
    if not v:
        return f"  {label}: n=0"
    c = Counter(v)
    total = sum(c.values())
    parts = [f"{k}:{100*n/total:.1f}%" for k, n in c.most_common(top)]
    return f"  {label} (n={len(v)}): " + "  ".join(parts)

def bool_pct(values, label=""):
    v = [x for x in values if x is not None]
    if not v:
        return f"  {label}: n=0"
    return f"  {label} (n={len(v)}): {100*sum(v)/len(v):.1f}%  ({sum(v)}/{len(v)})"

def compare(feat, label, fmt=".2f"):
    print(stats([r[feat] for r in clicks],     f"  click     {label}", fmt))
    print(stats([r[feat] for r in non_clicks], f"  non_click {label}", fmt))

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSE
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Coordonnées ────────────────────────────────────────────────────────────
print("═" * 70)
print("  1. COORDONNÉES GALACTIQUES")
print("═" * 70)
for axis in ("x", "y", "z"):
    print(f"\n  Axe {axis.upper()} :")
    compare(axis, axis.upper(), ".1f")

# ── 2. Étoile principale ──────────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  2. ÉTOILE PRINCIPALE")
print("═" * 70)

print("\n  Type :")
print(freq([r["primary_star_type"] for r in clicks],     "  click    "))
print(freq([r["primary_star_type"] for r in non_clicks], "  non_click"))

print("\n  Sous-type (Subclass) :")
print(freq([r["primary_subclass"] for r in clicks],     "  click    "))
print(freq([r["primary_subclass"] for r in non_clicks], "  non_click"))

print("\n  Masse stellaire (masses solaires) :")
compare("stellar_mass", "masse")

print("\n  Température de surface (K) :")
compare("surface_temp", "temp K", ".0f")

print("\n  Luminosité :")
print(freq([r["luminosity"] for r in clicks],     "  click    "))
print(freq([r["luminosity"] for r in non_clicks], "  non_click"))

print("\n  Âge (My) :")
compare("star_age_my", "âge My", ".0f")

print("\n  Ceinture autour étoile principale :")
print(bool_pct([r["has_belt"] for r in clicks],     "  click    "))
print(bool_pct([r["has_belt"] for r in non_clicks], "  non_click"))

# ── 3. Composition du système ─────────────────────────────────────────────────
print("\n" + "═" * 70)
print("  3. COMPOSITION DU SYSTÈME")
print("═" * 70)

print("\n  Nombre de corps total (FSS) :")
compare("body_count", "corps")

print("\n  Nombre d'étoiles :")
compare("stars_count", "étoiles")

print("\n  Nombre de planètes :")
compare("planets_count", "planètes")

print("\n  Atterrissables :")
compare("landables", "landables")

print("\n  Terraformables :")
compare("n_terraformables", "terraform")

print("\n  Géante gazeuse présente :")
print(bool_pct([r["has_gas_giant"]   for r in clicks],     "  click    "))
print(bool_pct([r["has_gas_giant"]   for r in non_clicks], "  non_click"))

print("\n  Corps avec anneaux :")
print(bool_pct([r["has_ringed_body"] for r in clicks],     "  click    "))
print(bool_pct([r["has_ringed_body"] for r in non_clicks], "  non_click"))

# ── 4. Signaux biologiques/géologiques ───────────────────────────────────────
print("\n" + "═" * 70)
print("  4. SIGNAUX FSS  [non_click uniquement — non disponible côté click]")
print("═" * 70)

print(stats([r["n_bio_signals"] for r in non_clicks], "\n  non_click bio_signals"))
print(stats([r["n_geo_signals"] for r in non_clicks], "  non_click geo_signals"))

# ── 5. Export CSV ─────────────────────────────────────────────────────────────
print("\n" + "═" * 70)

fields = [
    "name", "label", "x", "y", "z",
    "body_count", "stars_count", "planets_count", "landables",
    "has_gas_giant", "has_ringed_body", "n_terraformables",
    "primary_star_type", "primary_subclass", "stellar_mass",
    "surface_temp", "luminosity", "star_age_my", "has_belt",
    "n_stars_in_system", "n_bio_signals", "n_geo_signals",
]

with open("systems_dataset_v2.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in records:
        w.writerow({k: r.get(k) for k in fields})

print(f"\n✓ Dataset exporté : systems_dataset_v2.csv  ({len(records)} lignes, {len(fields)} colonnes)")
