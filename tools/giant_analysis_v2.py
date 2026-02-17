"""
giant_analysis_v2.py
--------------------
Analyse rigoureuse des géantes gazeuses et leur régime dynamique/thermique
dans les systèmes click vs non_click.

Méthode :
- Luminosité via AbsoluteMagnitude : L/L☉ = 10^((M☉ - Mv)/2.5)
- Flux relatif : S = (L/L☉) / a_AU^2  (Terre = 1)
- Frost line : r_frost ≈ 2.7 * sqrt(L/L☉) AU
- Position : ρ = a_AU / r_frost

Usage:
    python giant_analysis_v2.py
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats as scipy_stats

# ── Constantes ────────────────────────────────────────────────────────────────

AU = 1.495978707e11  # mètres
M_SUN = 4.83  # magnitude absolue du soleil

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

# ── Calculs physiques ─────────────────────────────────────────────────────────

def luminosity_from_abs_mag(abs_mag):
    """
    Calcule L/L☉ depuis magnitude absolue.
    L/L☉ = 10^((M☉ - Mv)/2.5)
    """
    if abs_mag is None:
        return None
    return 10 ** ((M_SUN - abs_mag) / 2.5)

def flux_relative_to_earth(L_ratio, semi_major_au):
    """
    Flux reçu par la planète relativement à la Terre.
    S = (L/L☉) / a_AU^2
    """
    if L_ratio is None or semi_major_au is None or semi_major_au <= 0:
        return None
    return L_ratio / (semi_major_au ** 2)

def frost_line_au(L_ratio):
    """
    Frost line approximative.
    r_frost ≈ 2.7 * sqrt(L/L☉) AU
    """
    if L_ratio is None or L_ratio <= 0:
        return None
    return 2.7 * np.sqrt(L_ratio)

def relative_to_frost(semi_major_au, frost_au):
    """
    Position relative à la frost line.
    ρ = a / r_frost
    """
    if semi_major_au is None or frost_au is None or frost_au <= 0:
        return None
    return semi_major_au / frost_au

# ── Extraction géantes ────────────────────────────────────────────────────────

def extract_giants_metadata(path):
    """Parse metadata (click) et extrait géantes + paramètres orbitaux."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    
    system_name = d["context"]["system"]["name"]
    stars = d.get("bodies", {}).get("stars", [])
    planets = d.get("bodies", {}).get("planets", [])
    
    # Étoile principale
    primary = None
    for s in stars:
        if s.get("DistanceFromArrivalLS", 999) == 0.0:
            primary = s
            break
    if not primary and stars:
        primary = stars[0]
    
    if not primary:
        return system_name, None, []
    
    abs_mag = primary.get("AbsoluteMagnitude")
    L_ratio = luminosity_from_abs_mag(abs_mag)
    frost_au = frost_line_au(L_ratio)
    
    giants = []
    for p in planets:
        if not is_gas_giant(p.get("PlanetClass")):
            continue
        
        semi_major_m = p.get("SemiMajorAxis")
        if semi_major_m is None or semi_major_m <= 0:
            continue
        
        semi_major_au = semi_major_m / AU
        S = flux_relative_to_earth(L_ratio, semi_major_au)
        rho = relative_to_frost(semi_major_au, frost_au)
        
        giants.append({
            "planet_name": p.get("BodyName"),
            "a_AU": semi_major_au,
            "S_earth": S,
            "rho_frost": rho,
            "mass_em": p.get("MassEM"),
        })
    
    return system_name, {"L_ratio": L_ratio, "frost_au": frost_au}, giants

def extract_giants_non_click_v2(path):
    """Parse non_click v2 et extrait géantes + paramètres orbitaux."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    
    system_name = d["system"]["name"]
    pstar = d.get("primary_star", {})
    planets = d.get("bodies", {}).get("planets", [])
    
    abs_mag = pstar.get("AbsoluteMagnitude")
    L_ratio = luminosity_from_abs_mag(abs_mag)
    frost_au = frost_line_au(L_ratio)
    
    if not L_ratio:
        return system_name, None, []
    
    giants = []
    for p in planets:
        if not is_gas_giant(p.get("PlanetClass")):
            continue
        
        semi_major_m = p.get("SemiMajorAxis")
        if semi_major_m is None or semi_major_m <= 0:
            continue
        
        semi_major_au = semi_major_m / AU
        S = flux_relative_to_earth(L_ratio, semi_major_au)
        rho = relative_to_frost(semi_major_au, frost_au)
        
        giants.append({
            "planet_name": p.get("BodyName"),
            "a_AU": semi_major_au,
            "S_earth": S,
            "rho_frost": rho,
            "mass_em": p.get("MassEM"),
        })
    
    return system_name, {"L_ratio": L_ratio, "frost_au": frost_au}, giants

# ── Chargement ────────────────────────────────────────────────────────────────

systems = []  # {name, label, star_params, giants[]}

for p in Path("metadata").glob("*.json"):
    try:
        name, star, giants = extract_giants_metadata(p)
        systems.append({"name": name, "label": "click", "star": star, "giants": giants})
    except Exception as e:
        print(f"Erreur metadata {p.name}: {e}")

for p in Path("systems/non_click").glob("*.json"):
    try:
        name, star, giants = extract_giants_non_click_v2(p)
        systems.append({"name": name, "label": "non_click", "star": star, "giants": giants})
    except Exception as e:
        print(f"Erreur non_click {p.name}: {e}")

# Filtrer systèmes avec géantes
systems_with_giants = [s for s in systems if s["giants"]]

click_sys = [s for s in systems_with_giants if s["label"] == "click"]
nc_sys = [s for s in systems_with_giants if s["label"] == "non_click"]

print(f"Systèmes avec géantes : {len(click_sys)} click, {len(nc_sys)} non_click\n")

# Total géantes
n_giants_click = sum(len(s["giants"]) for s in click_sys)
n_giants_nc = sum(len(s["giants"]) for s in nc_sys)
print(f"Géantes détectées : {n_giants_click} click, {n_giants_nc} non_click\n")

# ── Features au niveau système ────────────────────────────────────────────────

def compute_system_features(system):
    """Calcule features agrégées au niveau système."""
    giants = system["giants"]
    if not giants:
        return None
    
    a_vals = [g["a_AU"] for g in giants if g["a_AU"]]
    S_vals = [g["S_earth"] for g in giants if g["S_earth"]]
    rho_vals = [g["rho_frost"] for g in giants if g["rho_frost"]]
    
    # Géantes dans zone frost (0.7 < rho < 1.3)
    n_near_frost = sum(1 for rho in rho_vals if 0.7 <= rho <= 1.3)
    
    # Géantes chaudes (rho < 1.0, migration)
    n_migrated = sum(1 for rho in rho_vals if rho < 1.0)
    
    # Géantes très chaudes (S > 1.0, plus irradiées que la Terre)
    n_hot = sum(1 for S in S_vals if S > 1.0)
    
    return {
        "min_a_AU": min(a_vals) if a_vals else None,
        "max_S_earth": max(S_vals) if S_vals else None,
        "min_rho_frost": min(rho_vals) if rho_vals else None,
        "n_giants": len(giants),
        "n_near_frost": n_near_frost,
        "n_migrated": n_migrated,
        "n_hot": n_hot,
    }

# Calcul features
for s in systems_with_giants:
    s["features"] = compute_system_features(s)

# ── Analyse comparative ───────────────────────────────────────────────────────

def get_values(systems, key):
    """Extrait valeurs d'une feature depuis liste de systèmes."""
    return [s["features"][key] for s in systems 
            if s["features"] and s["features"][key] is not None]

def stats_str(values, label):
    v = [x for x in values if x is not None and not np.isnan(x) and not np.isinf(x)]
    if not v:
        return f"{label}: n=0"
    return (f"{label} (n={len(v)}): "
            f"mean={np.mean(v):.3f}  median={np.median(v):.3f}  "
            f"[{min(v):.3f} – {max(v):.3f}]")

def mann_whitney(click_vals, nc_vals, label):
    """Test Mann-Whitney U."""
    if len(click_vals) < 3 or len(nc_vals) < 3:
        return None
    u, p = scipy_stats.mannwhitneyu(click_vals, nc_vals, alternative='two-sided')
    return (label, p, f"médiane click={np.median(click_vals):.3f} vs nc={np.median(nc_vals):.3f}")

print("=" * 80)
print("  DISTANCE ORBITALE MINIMALE (UA) — géante la plus proche")
print("=" * 80)
c_min_a = get_values(click_sys, "min_a_AU")
n_min_a = get_values(nc_sys, "min_a_AU")
print(stats_str(c_min_a, "  click    "))
print(stats_str(n_min_a, "  non_click"))

print("\n" + "=" * 80)
print("  FLUX MAXIMAL (S/S_earth) — géante la plus irradiée")
print("=" * 80)
c_max_S = get_values(click_sys, "max_S_earth")
n_max_S = get_values(nc_sys, "max_S_earth")
print(stats_str(c_max_S, "  click    "))
print(stats_str(n_max_S, "  non_click"))

print("\n" + "=" * 80)
print("  POSITION RELATIVE À FROST LINE (ρ) — géante la plus proche")
print("=" * 80)
c_min_rho = get_values(click_sys, "min_rho_frost")
n_min_rho = get_values(nc_sys, "min_rho_frost")
print(stats_str(c_min_rho, "  click    "))
print(stats_str(n_min_rho, "  non_click"))
print("\n  ρ < 1.0 : migration (en dedans de frost line)")
print("  ρ ≈ 1.0 : formation in-situ")
print("  ρ > 1.0 : zone normale (au-delà)")

print("\n" + "=" * 80)
print("  FRÉQUENCE DE CONFIGURATIONS SPÉCIFIQUES")
print("=" * 80)

# % systèmes avec au moins une géante migrée
c_migrated_pct = 100 * sum(1 for s in click_sys if s["features"]["n_migrated"] > 0) / len(click_sys) if click_sys else 0
n_migrated_pct = 100 * sum(1 for s in nc_sys if s["features"]["n_migrated"] > 0) / len(nc_sys) if nc_sys else 0
print(f"  Systèmes avec géante(s) migrée(s) (ρ < 1.0) :")
print(f"    click:     {c_migrated_pct:.1f}%  ({sum(1 for s in click_sys if s['features']['n_migrated'] > 0)}/{len(click_sys)})")
if nc_sys:
    print(f"    non_click: {n_migrated_pct:.1f}%  ({sum(1 for s in nc_sys if s['features']['n_migrated'] > 0)}/{len(nc_sys)})")
else:
    print(f"    non_click: N/A (aucun système avec géante)")

# % systèmes avec géante proche frost line
c_near_pct = 100 * sum(1 for s in click_sys if s["features"]["n_near_frost"] > 0) / len(click_sys) if click_sys else 0
n_near_pct = 100 * sum(1 for s in nc_sys if s["features"]["n_near_frost"] > 0) / len(nc_sys) if nc_sys else 0
print(f"\n  Systèmes avec géante(s) proche frost line (0.7 < ρ < 1.3) :")
print(f"    click:     {c_near_pct:.1f}%  ({sum(1 for s in click_sys if s['features']['n_near_frost'] > 0)}/{len(click_sys)})")
if nc_sys:
    print(f"    non_click: {n_near_pct:.1f}%  ({sum(1 for s in nc_sys if s['features']['n_near_frost'] > 0)}/{len(nc_sys)})")
else:
    print(f"    non_click: N/A (aucun système avec géante)")

# % systèmes avec géante chaude (S > 1)
c_hot_pct = 100 * sum(1 for s in click_sys if s["features"]["n_hot"] > 0) / len(click_sys) if click_sys else 0
n_hot_pct = 100 * sum(1 for s in nc_sys if s["features"]["n_hot"] > 0) / len(nc_sys) if nc_sys else 0
print(f"\n  Systèmes avec géante(s) très irradiée(s) (S > Terre) :")
print(f"    click:     {c_hot_pct:.1f}%  ({sum(1 for s in click_sys if s['features']['n_hot'] > 0)}/{len(click_sys)})")
if nc_sys:
    print(f"    non_click: {n_hot_pct:.1f}%  ({sum(1 for s in nc_sys if s['features']['n_hot'] > 0)}/{len(nc_sys)})")
else:
    print(f"    non_click: N/A (aucun système avec géante)")

# ── Tests statistiques ────────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("  TESTS STATISTIQUES (Mann-Whitney U)")
print("=" * 80)

tests = [
    mann_whitney(c_min_a, n_min_a, "Distance minimale (a_AU)"),
    mann_whitney(c_max_S, n_max_S, "Flux maximal (S_earth)"),
    mann_whitney(c_min_rho, n_min_rho, "Position minimale (ρ_frost)"),
]

alpha_bonf = 0.05 / len([t for t in tests if t])

print(f"\nCorrection Bonferroni : alpha = 0.05 / {len([t for t in tests if t])} = {alpha_bonf:.4f}\n")
print(f"{'Feature':<30} {'p-value':<12} {'Signif':<6} {'Détail'}")
print("-" * 80)

for t in tests:
    if t is None:
        continue
    label, p, detail = t
    sig = "***" if p < alpha_bonf else ("*" if p < 0.05 else "")
    print(f"{label:<30} {p:<12.4f} {sig:<6} {detail}")

print("\n*** = significatif après Bonferroni")
print("*   = tendance (p < 0.05)")

# ── Export ────────────────────────────────────────────────────────────────────

import csv
rows = []
for s in systems_with_giants:
    for g in s["giants"]:
        rows.append({
            "system": s["name"],
            "label": s["label"],
            "planet": g["planet_name"],
            "a_AU": g["a_AU"],
            "S_earth": g["S_earth"],
            "rho_frost": g["rho_frost"],
            "mass_em": g["mass_em"],
        })

with open("giants_analysis_v2.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["system","label","planet","a_AU","S_earth","rho_frost","mass_em"])
    w.writeheader()
    for row in rows:
        w.writerow(row)

print(f"\n✓ Export : giants_analysis_v2.csv ({len(rows)} géantes)")

