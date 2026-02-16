"""
extract_non_click_v2.py
-----------------------
Parse les journaux Elite Dangerous (février 2026+) et génère des fichiers
JSON enrichis pour les systèmes non_click, dans systems/non_click_v2/.

Usage :
    python extract_non_click_v2.py

Ajuste les chemins en haut du fichier si besoin.
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

# ── Configuration ─────────────────────────────────────────────────────────────

LOG_DIR      = Path(r"C:\Users\Max\Saved Games\Frontier Developments\Elite Dangerous")
NON_CLICK_V1 = Path("systems/non_click")          # pour récupérer la liste des systèmes connus
OUTPUT_DIR   = Path("systems/non_click_v2")
CUTOFF       = datetime(2026, 2, 1, tzinfo=timezone.utc)

# ── Chargement de la liste des systèmes non_click connus ─────────────────────

known_non_click = {}   # address (int) → nom
for p in NON_CLICK_V1.glob("*.json"):
    try:
        with open(p) as f:
            d = json.load(f)
        addr = d["system"]["address"]
        known_non_click[addr] = d
    except Exception as e:
        print(f"  SKIP v1 {p.name}: {e}")

print(f"Systèmes non_click connus : {len(known_non_click)}")

# ── Chargement des logs (février 2026+) ──────────────────────────────────────

def log_date(filename):
    """Extrait la date depuis le nom Journal.YYYY-MM-DDTHHMMss.nn.log"""
    m = re.search(r"Journal\.(\d{4}-\d{2}-\d{2})", filename)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return None

log_files = sorted(
    [p for p in LOG_DIR.glob("Journal.*.log")
     if (d := log_date(p.name)) and d >= CUTOFF],
    key=lambda p: p.name
)
print(f"Fichiers log à parser : {len(log_files)}")

# ── Parser les logs ───────────────────────────────────────────────────────────

# Structure : system_data[address] = { ... tout ce qu'on a collecté ... }
system_data = defaultdict(lambda: {
    "name": None,
    "address": None,
    "star_pos": None,
    "log_files": set(),
    "first_seen": None,
    "last_seen": None,
    "fss_body_count": None,
    "fss_non_body_count": None,
    "stars": [],
    "planets": [],
    "fss_body_signals": [],   # liste de {body_name, signals}
    "saa_mappings": [],       # corps mappés
    "codex_entries": [],
})

current_system_addr = None

for log_path in log_files:
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  Erreur lecture {log_path.name}: {e}")
        continue

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except Exception:
            continue

        event = ev.get("event")
        ts_str = ev.get("timestamp", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            ts = None

        # ── Nouveau système ──────────────────────────────────────────────────
        if event == "FSDJump":
            addr = ev.get("SystemAddress")
            if addr is None:
                continue
            current_system_addr = addr
            sd = system_data[addr]
            sd["name"]     = ev.get("StarSystem")
            sd["address"]  = addr
            sd["star_pos"] = ev.get("StarPos")
            sd["log_files"].add(log_path.name)
            if ts:
                if sd["first_seen"] is None or ts < sd["first_seen"]:
                    sd["first_seen"] = ts
                if sd["last_seen"] is None or ts > sd["last_seen"]:
                    sd["last_seen"] = ts

        if current_system_addr is None:
            continue

        # Vérification systématique : on ignore tout event sans SystemAddress
        # correspondant au système courant (sauf exceptions documentées ci-dessous)
        ev_addr = ev.get("SystemAddress")

        # Certains events n'ont pas de SystemAddress (ex: Music, Fileheader...)
        # Pour ceux qui DOIVENT en avoir, on vérifie strictement.
        EVENTS_WITH_ADDR = {
            "Scan", "FSSDiscoveryScan", "FSSBodySignals",
            "FSSAllBodiesFound", "SAAScanComplete", "CodexEntry",
        }
        if event in EVENTS_WITH_ADDR and ev_addr != current_system_addr:
            continue

        sd = system_data[current_system_addr]

        # ── Mise à jour timestamp ────────────────────────────────────────────
        if ts and (ev_addr == current_system_addr or ev_addr is None):
            if sd["last_seen"] is None or ts > sd["last_seen"]:
                sd["last_seen"] = ts

        # ── FSS discovery ────────────────────────────────────────────────────
        if event == "FSSDiscoveryScan":
            sd["fss_body_count"]     = ev.get("BodyCount")
            sd["fss_non_body_count"] = ev.get("NonBodyCount")

        # ── Scan corps ───────────────────────────────────────────────────────
        elif event == "Scan":
            if "StarType" in ev:
                star = {
                    "BodyName":              ev.get("BodyName"),
                    "BodyID":                ev.get("BodyID"),
                    "DistanceFromArrivalLS": ev.get("DistanceFromArrivalLS"),
                    "StarType":              ev.get("StarType"),
                    "Subclass":              ev.get("Subclass"),
                    "StellarMass":           ev.get("StellarMass"),
                    "Radius":                ev.get("Radius"),
                    "AbsoluteMagnitude":     ev.get("AbsoluteMagnitude"),
                    "Age_MY":                ev.get("Age_MY"),
                    "SurfaceTemperature":    ev.get("SurfaceTemperature"),
                    "Luminosity":            ev.get("Luminosity"),
                    "RotationPeriod":        ev.get("RotationPeriod"),
                    "Rings":                 ev.get("Rings"),
                }
                existing_ids = {s["BodyID"] for s in sd["stars"]}
                if star["BodyID"] not in existing_ids:
                    sd["stars"].append(star)
            elif "PlanetClass" in ev:
                planet = {
                    "BodyName":              ev.get("BodyName"),
                    "BodyID":                ev.get("BodyID"),
                    "DistanceFromArrivalLS": ev.get("DistanceFromArrivalLS"),
                    "PlanetClass":           ev.get("PlanetClass"),
                    "TerraformState":        ev.get("TerraformState"),
                    "AtmosphereType":        ev.get("AtmosphereType"),
                    "Volcanism":             ev.get("Volcanism"),
                    "MassEM":                ev.get("MassEM"),
                    "Landable":              ev.get("Landable"),
                    "Rings":                 ev.get("Rings"),
                    "WasDiscovered":         ev.get("WasDiscovered"),
                    "WasMapped":             ev.get("WasMapped"),
                }
                existing_ids = {p["BodyID"] for p in sd["planets"]}
                if planet["BodyID"] not in existing_ids:
                    sd["planets"].append(planet)

        # ── Signaux FSS ──────────────────────────────────────────────────────
        elif event == "FSSBodySignals":
            sd["fss_body_signals"].append({
                "BodyName": ev.get("BodyName"),
                "Signals":  ev.get("Signals", []),
            })

        # ── Mappings SAA ─────────────────────────────────────────────────────
        elif event == "SAAScanComplete":
            sd["saa_mappings"].append(ev.get("BodyName"))

        # ── Codex ────────────────────────────────────────────────────────────
        elif event == "CodexEntry":
            sd["codex_entries"].append({
                "Name":        ev.get("Name_Localised", ev.get("Name")),
                "Category":    ev.get("Category_Localised", ev.get("Category")),
                "SubCategory": ev.get("SubCategory_Localised", ev.get("SubCategory")),
                "IsNewEntry":  ev.get("IsNewEntry"),
            })

print(f"Systèmes parsés depuis les logs : {len(system_data)}")

# ── Filtrage : garder uniquement les non_click connus ────────────────────────

matched   = {addr: sd for addr, sd in system_data.items() if addr in known_non_click}
unmatched = len(system_data) - len(matched)
print(f"Matched avec non_click v1 : {len(matched)}  (non matchés ignorés : {unmatched})")

# ── Export JSON ───────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def primary_star(stars):
    """Étoile principale = DistanceFromArrivalLS == 0, sinon la première."""
    for s in stars:
        if s.get("DistanceFromArrivalLS", 999) == 0.0:
            return s
    return stars[0] if stars else {}

written = 0
for addr, sd in matched.items():
    v1 = known_non_click[addr]
    v1_feat = v1.get("features", {})
    v1_bodies = v1_feat.get("bodies", {})
    v1_stars  = v1_feat.get("stars", {})

    stars_list   = sd["stars"]
    planets_list = sd["planets"]
    pstar        = primary_star(stars_list)

    # features enrichies
    planet_classes = list({p["PlanetClass"] for p in planets_list if p.get("PlanetClass")})

    GAS_GIANT_CLASSES = {
        "gas giant with water based life", "gas giant with ammonia based life",
        "sudarsky class i gas giant", "sudarsky class ii gas giant",
        "sudarsky class iii gas giant", "sudarsky class iv gas giant",
        "sudarsky class v gas giant", "helium rich gas giant",
        "helium gas giant", "water giant",
    }
    has_gas_giant = any(
        p.get("PlanetClass", "").lower() in GAS_GIANT_CLASSES or
        "giant" in p.get("PlanetClass", "").lower()
        for p in planets_list
    )
    has_ringed     = any(p.get("Rings") for p in planets_list + stars_list)
    n_terraformables = sum(1 for p in planets_list
                           if p.get("TerraformState") not in (None, "", "None"))
    n_landables    = sum(1 for p in planets_list if p.get("Landable"))
    star_types     = list({s["StarType"] for s in stars_list if s.get("StarType")})
    n_bio_signals  = sum(
        sum(sig.get("Count", 0) for sig in body["Signals"]
            if "Biological" in sig.get("Type_Localised","") or "Biological" in sig.get("Type",""))
        for body in sd["fss_body_signals"]
    )
    n_geo_signals  = sum(
        sum(sig.get("Count", 0) for sig in body["Signals"]
            if "Geological" in sig.get("Type_Localised","") or "Geological" in sig.get("Type",""))
        for body in sd["fss_body_signals"]
    )

    out = {
        "schema_version": 2,
        "label": "non_click",
        "provenance": {
            "generated_by": "extract_non_click_v2.py",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "log_files": sorted(sd["log_files"]),
        },
        "system": {
            "name":     sd["name"] or v1["system"]["name"],
            "address":  addr,
            "star_pos": sd["star_pos"] or v1["system"]["star_pos"],
        },
        "evidence": {
            "first_seen_utc": sd["first_seen"].isoformat() if sd["first_seen"] else
                              v1.get("evidence",{}).get("journal",{}).get("first_seen_utc"),
            "last_seen_utc":  sd["last_seen"].isoformat()  if sd["last_seen"]  else
                              v1.get("evidence",{}).get("journal",{}).get("last_seen_utc"),
        },
        "features": {
            "bodies": {
                "fss_bodycount":      sd["fss_body_count"]     or v1_bodies.get("fss_bodycount"),
                "fss_nonbodycount":   sd["fss_non_body_count"] or v1_bodies.get("fss_nonbodycount"),
                "n_bodies_scanned":   len(stars_list) + len(planets_list) if (stars_list or planets_list) else None,
                "n_stars_scanned":    len(stars_list)   if stars_list   else None,
                "n_planets_scanned":  len(planets_list) if planets_list else None,
                "n_landables":        n_landables if planets_list else None,
                "has_gas_giant":      has_gas_giant,
                "has_ringed_body":    has_ringed,
                "n_terraformables":   n_terraformables,
                "planet_classes":     planet_classes,
            },
            "stars": {
                "n_stars":    len(stars_list) or v1_stars.get("n_stars"),
                "types":      star_types or v1_stars.get("types", []),
            },
            "signals": {
                "n_fss_body_signals_events": len(sd["fss_body_signals"]),
                "n_bio_signals":  n_bio_signals,
                "n_geo_signals":  n_geo_signals,
                "n_saa_mappings": len(sd["saa_mappings"]),
            },
            "codex": {
                "n_entries": len(sd["codex_entries"]),
            },
        },
        "primary_star": pstar if pstar else None,
        "bodies": {
            "stars":   stars_list,
            "planets": planets_list,
        },
        "fss_body_signals": sd["fss_body_signals"],
        "saa_mappings":     sd["saa_mappings"],
        "codex_entries":    sd["codex_entries"],
    }

    # nom de fichier = même convention que v1
    safe_name = re.sub(r"[^\w\-]", "_", out["system"]["name"])
    out_path  = OUTPUT_DIR / f"{safe_name}__{addr}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=str)
    written += 1

print(f"\n✓ {written} fichiers écrits dans {OUTPUT_DIR}/")

# ── Rapport rapide ────────────────────────────────────────────────────────────

n_with_stars   = sum(1 for sd in matched.values() if sd["stars"])
n_with_planets = sum(1 for sd in matched.values() if sd["planets"])
n_no_data      = sum(1 for sd in matched.values() if not sd["stars"] and not sd["planets"])

print(f"\nRapport :")
print(f"  Systèmes avec étoile(s) scannée(s)  : {n_with_stars}")
print(f"  Systèmes avec planète(s) scannée(s) : {n_with_planets}")
print(f"  Systèmes sans scan détaillé          : {n_no_data}  (données v1 conservées)")
