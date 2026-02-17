"""
pattern_mining.py
-----------------
Recherche de patterns, combinaisons rares et structures répétées
qui distinguent les systèmes click des non_click.

Usage:
    python pattern_mining.py
"""

import csv
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

# ── Chargement ────────────────────────────────────────────────────────────────

rows = []
with open("systems_dataset_v2.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)

clicks     = [r for r in rows if r["label"] == "click"]
non_clicks = [r for r in rows if r["label"] == "non_click"]

print(f"Dataset: {len(clicks)} click, {len(non_clicks)} non_click\n")

# ── Helpers ───────────────────────────────────────────────────────────────────

def discretize(value, bins):
    """Discrétise une valeur continue en catégories."""
    try:
        v = float(value)
        for label, (low, high) in bins.items():
            if low <= v < high:
                return label
        return bins.get("other", "unknown")
    except:
        return "unknown"

def extract_features(row):
    """Extrait features discrétisées pour pattern mining."""
    feat = {}
    
    # Type et subclass étoile
    if row["primary_star_type"] not in ("", "None"):
        feat["type"] = row["primary_star_type"]
    if row["primary_subclass"] not in ("", "None"):
        feat["subclass"] = row["primary_subclass"]
    
    # Masse stellaire
    feat["mass"] = discretize(row["stellar_mass"], {
        "dwarf":  (0.0, 0.5),
        "medium": (0.5, 1.2),
        "heavy":  (1.2, 10.0),
    })
    
    # Température
    feat["temp"] = discretize(row["surface_temp"], {
        "cool":   (0, 3500),
        "warm":   (3500, 6000),
        "hot":    (6000, 30000),
    })
    
    # Âge
    feat["age"] = discretize(row["star_age_my"], {
        "young": (0, 3000),
        "mid":   (3000, 8000),
        "old":   (8000, 20000),
    })
    
    # Luminosité
    if row["luminosity"] not in ("", "None"):
        feat["lum"] = row["luminosity"]
    
    # Booléens
    feat["has_gas_giant"]   = row["has_gas_giant"] == "True"
    feat["has_ringed_body"] = row["has_ringed_body"] == "True"
    feat["has_belt"]        = row["has_belt"] == "True"
    
    # Nombre d'étoiles
    feat["n_stars"] = discretize(row["stars_count"], {
        "single": (1, 1.5),
        "multi":  (1.5, 10),
    })
    
    # Nombre de corps
    feat["body_count"] = discretize(row["body_count"], {
        "small":  (0, 8),
        "medium": (8, 15),
        "large":  (15, 100),
    })
    
    # Coordonnées discrétisées (pour clustering spatial)
    feat["x_zone"] = discretize(row["x"], {
        "low":  (0, 5000),
        "mid":  (5000, 7500),
        "high": (7500, 15000),
    })
    
    feat["z_zone"] = discretize(row["z"], {
        "negative": (-5000, 0),
        "low":      (0, 1000),
        "mid":      (1000, 2000),
        "high":     (2000, 5000),
    })
    
    return feat

# ── 1. COMBINAISONS RARES ─────────────────────────────────────────────────────

print("=" * 80)
print("  1. COMBINAISONS RARES (présentes dans clicks mais rares dans non_clicks)")
print("=" * 80)

click_feats = [extract_features(r) for r in clicks]
nc_feats    = [extract_features(r) for r in non_clicks]

# Génère toutes les paires de features
all_keys = set()
for f in click_feats + nc_feats:
    all_keys.update(f.keys())

rare_patterns = []

# Teste toutes les combinaisons de 2-3 features
for size in (2, 3):
    for keys in combinations(sorted(all_keys), size):
        # Compte occurrences dans clicks
        c_patterns = Counter()
        for f in click_feats:
            pattern = tuple((k, f.get(k)) for k in keys if k in f)
            if len(pattern) == size:
                c_patterns[pattern] += 1
        
        # Compte occurrences dans non_clicks
        nc_patterns = Counter()
        for f in nc_feats:
            pattern = tuple((k, f.get(k)) for k in keys if k in f)
            if len(pattern) == size:
                nc_patterns[pattern] += 1
        
        # Cherche patterns surreprésentés dans clicks
        for pattern, c_count in c_patterns.items():
            nc_count = nc_patterns.get(pattern, 0)
            c_freq = c_count / len(clicks)
            nc_freq = nc_count / len(non_clicks) if nc_count > 0 else 0.001
            
            # Ratio > 3 et présent dans au moins 20% des clicks
            if c_freq >= 0.2 and (c_freq / nc_freq) > 3:
                rare_patterns.append({
                    "pattern": pattern,
                    "click_freq": c_freq,
                    "nc_freq": nc_freq,
                    "ratio": c_freq / nc_freq,
                    "click_count": c_count,
                    "nc_count": nc_count,
                })

# Tri par ratio décroissant
rare_patterns.sort(key=lambda x: x["ratio"], reverse=True)

print(f"\nTop 10 patterns surreprésentés dans clicks :\n")
for i, p in enumerate(rare_patterns[:10], 1):
    pattern_str = " & ".join(f"{k}={v}" for k, v in p["pattern"])
    print(f"{i:2}. {pattern_str}")
    print(f"    click: {p['click_freq']*100:.1f}% ({p['click_count']}/{len(clicks)})  "
          f"non_click: {p['nc_freq']*100:.1f}% ({p['nc_count']}/{len(non_clicks)})  "
          f"ratio: {p['ratio']:.1f}x")

# ── 2. RÈGLES D'ASSOCIATION ───────────────────────────────────────────────────

print("\n" + "=" * 80)
print("  2. RÈGLES D'ASSOCIATION (si A et B alors click?)")
print("=" * 80)

def check_rule(condition_fn, data):
    """Compte combien d'éléments satisfont la condition."""
    return sum(1 for r in data if condition_fn(r))

rules = []

# Règle 1: type + masse + géante gazeuse
def rule1(r):
    return (r.get("type") in ("F", "A", "B", "G") and
            r.get("mass") == "heavy" and
            r.get("has_gas_giant") == True)

c1 = check_rule(rule1, click_feats)
n1 = check_rule(rule1, nc_feats)
if c1 > 0:
    rules.append(("Type chaud (F/A/B/G) + masse élevée + géante gazeuse", 
                  c1, n1, c1/len(clicks), n1/len(non_clicks)))

# Règle 2: subclass 9 + géante gazeuse
def rule2(r):
    return r.get("subclass") == "9" and r.get("has_gas_giant") == True

c2 = check_rule(rule2, click_feats)
n2 = check_rule(rule2, nc_feats)
if c2 > 0:
    rules.append(("Subclass 9 + géante gazeuse", 
                  c2, n2, c2/len(clicks), n2/len(non_clicks)))

# Règle 3: température chaude + corps annelés
def rule3(r):
    return r.get("temp") == "hot" and r.get("has_ringed_body") == True

c3 = check_rule(rule3, click_feats)
n3 = check_rule(rule3, nc_feats)
if c3 > 0:
    rules.append(("Température élevée + corps annelés", 
                  c3, n3, c3/len(clicks), n3/len(non_clicks)))

# Règle 4: zone spatiale spécifique
def rule4(r):
    return r.get("x_zone") == "mid" and r.get("z_zone") == "mid"

c4 = check_rule(rule4, click_feats)
n4 = check_rule(rule4, nc_feats)
if c4 > 0:
    rules.append(("Zone X mid + Z mid", 
                  c4, n4, c4/len(clicks), n4/len(non_clicks)))

# Règle 5: jeune + chaud + géante
def rule5(r):
    return (r.get("age") == "young" and 
            r.get("temp") in ("warm", "hot") and
            r.get("has_gas_giant") == True)

c5 = check_rule(rule5, click_feats)
n5 = check_rule(rule5, nc_feats)
if c5 > 0:
    rules.append(("Jeune + chaud + géante gazeuse", 
                  c5, n5, c5/len(clicks), n5/len(non_clicks)))

print(f"\nRègles candidates :\n")
for i, (rule, c, n, c_pct, n_pct) in enumerate(rules, 1):
    precision = c / (c + n) if (c + n) > 0 else 0
    recall = c / len(clicks)
    print(f"{i}. {rule}")
    print(f"   Clicks: {c}/{len(clicks)} ({c_pct*100:.1f}%)  "
          f"Non-clicks: {n}/{len(non_clicks)} ({n_pct*100:.1f}%)")
    print(f"   Précision: {precision*100:.1f}%  Rappel: {recall*100:.1f}%")

# ── 3. SYSTÈMES CLICK UNIQUES ─────────────────────────────────────────────────

print("\n" + "=" * 80)
print("  3. CONFIGURATIONS UNIQUES AUX CLICKS (jamais vues dans non_clicks)")
print("=" * 80)

# Cherche les combinaisons type + autre feature uniques
unique_configs = []

for c_row, c_feat in zip(clicks, click_feats):
    sig = (c_feat.get("type"), c_feat.get("subclass"), 
           c_feat.get("has_gas_giant"), c_feat.get("mass"))
    
    # Vérifie si cette signature existe dans non_clicks
    found = any(
        (nf.get("type"), nf.get("subclass"), 
         nf.get("has_gas_giant"), nf.get("mass")) == sig
        for nf in nc_feats
    )
    
    if not found:
        unique_configs.append((c_row["name"], sig))

if unique_configs:
    print(f"\n{len(unique_configs)} systèmes click avec configurations jamais vues :\n")
    for name, (typ, sub, gg, mass) in unique_configs[:10]:
        print(f"  {name}: type={typ} sub={sub} mass={mass} gas_giant={gg}")
else:
    print("\nAucune configuration strictement unique.")

# ── 4. CLUSTERING SPATIAL ─────────────────────────────────────────────────────

print("\n" + "=" * 80)
print("  4. CLUSTERING SPATIAL (clicks proches les uns des autres?)")
print("=" * 80)

def parse_coords(r):
    try:
        return np.array([float(r["x"]), float(r["y"]), float(r["z"])])
    except:
        return None

click_coords = np.array([parse_coords(r) for r in clicks if parse_coords(r) is not None])

if len(click_coords) > 1:
    # Calcule distances moyennes intra-cluster
    distances = []
    for i in range(len(click_coords)):
        for j in range(i+1, len(click_coords)):
            dist = np.linalg.norm(click_coords[i] - click_coords[j])
            distances.append(dist)
    
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    min_dist = np.min(distances)
    max_dist = np.max(distances)
    
    print(f"\nDistances entre systèmes click :")
    print(f"  Moyenne:  {mean_dist:.1f} ly")
    print(f"  Médiane:  {median_dist:.1f} ly")
    print(f"  Min:      {min_dist:.1f} ly")
    print(f"  Max:      {max_dist:.1f} ly")
    
    # Cherche clusters (distance < 500 ly)
    clusters = []
    used = set()
    for i in range(len(click_coords)):
        if i in used:
            continue
        cluster = [i]
        for j in range(len(click_coords)):
            if j != i and j not in used:
                if np.linalg.norm(click_coords[i] - click_coords[j]) < 500:
                    cluster.append(j)
                    used.add(j)
        if len(cluster) > 1:
            clusters.append(cluster)
            used.add(i)
    
    if clusters:
        print(f"\n{len(clusters)} clusters détectés (distance < 500 ly) :")
        for i, cluster in enumerate(clusters, 1):
            names = [clicks[idx]["name"] for idx in cluster]
            print(f"  Cluster {i}: {len(cluster)} systèmes")
            for n in names[:3]:
                print(f"    - {n}")

print("\n" + "=" * 80)
