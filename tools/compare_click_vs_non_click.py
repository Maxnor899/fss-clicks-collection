#!/usr/bin/env python3
"""
compare_click_vs_non_click.py

Compares CLICK vs NON-CLICK systems and generates multiple plots (PNG) in figures/.

Data sources:
  - CLICK systems: analysis/summary.json (produced by analyze_clicks.py)
      Expected fields per system include: system_name, system_address, star_pos, Ii_db_median...
  - CLICK enrichment (optional but recommended): metadata/*.json
      We DO NOT write any click JSONs to systems/; we just read metadata directly to avoid repo pollution.
      We infer the system name from the raw journal events inside each metadata file (StarSystem/SystemName).
  - NON-CLICK systems: systems/non_click/*.json (produced by extract_non_click_systems.py)

Outputs:
  - figures/click_vs_nonclick_*.png

Usage (from repo root):
  python tools/compare_click_vs_non_click.py
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

CLICK_SUMMARY = Path("analysis/summary.json")
METADATA_DIR = Path("metadata")
NONCLICK_DIR = Path("systems/non_click")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _safe_pos(p: Any) -> Optional[Tuple[float, float, float]]:
    if isinstance(p, list) and len(p) == 3:
        try:
            return (float(p[0]), float(p[1]), float(p[2]))
        except Exception:
            return None
    return None


def _dist3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _infer_system_name_from_metadata(md: Dict[str, Any]) -> Optional[str]:
    """Infer Elite system name from our metadata JSON.

    Known places (in your repo):
      - md["context"]["system"]["name"]
      - md["raw"][...]["StarSystem"] / ["SystemName"]
    """
    # 1) Preferred: context.system.name
    ctx = md.get("context")
    if isinstance(ctx, dict):
        sys = ctx.get("system")
        if isinstance(sys, dict):
            name = sys.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
        # Sometimes flattened
        name = ctx.get("system_name") or ctx.get("StarSystem")
        if isinstance(name, str) and name.strip():
            return name.strip()

    # 2) Fallback: scan raw journal events
    raw = md.get("raw")
    if isinstance(raw, list):
        for ev in raw:
            if not isinstance(ev, dict):
                continue
            name = ev.get("StarSystem") or ev.get("SystemName") or ev.get("StarSystemName")
            if isinstance(name, str) and name.strip():
                return name.strip()

    return None


def load_metadata_index_by_name() -> Dict[str, Dict[str, Any]]:
    """Index metadata by inferred system name."""
    idx: Dict[str, Dict[str, Any]] = {}
    if not METADATA_DIR.exists():
        return idx
    for fp in METADATA_DIR.glob("*.json"):
        try:
            md = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        name = _infer_system_name_from_metadata(md)
        if not name:
            continue
        # keep first occurrence; duplicates are rare and usually equivalent
        idx.setdefault(name, md)
    return idx


def load_click_from_summary() -> List[Dict[str, Any]]:
    data = json.loads(CLICK_SUMMARY.read_text(encoding="utf-8"))
    rows: List[Dict[str, Any]] = []
    for s in data.get("systems", []):
        # Your summary.json uses system_name (not system)
        name = s.get("system_name") or s.get("system") or s.get("name")
        pos = _safe_pos(s.get("star_pos"))
        if not isinstance(name, str) or not name.strip() or not pos:
            continue
        rows.append({
            "label": "click",
            "name": name.strip(),
            "address": s.get("system_address"),
            "pos": pos,
            "Ii_db": s.get("Ii_db_median"),
        })
    return rows


def enrich_click_with_metadata(click_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adds body_count, stars_count, planets_count, gas giant and rings flags if metadata exists."""
    idx = load_metadata_index_by_name()
    out: List[Dict[str, Any]] = []
    for r in click_rows:
        md = idx.get(r["name"])
        body_count = stars_count = planets_count = None
        has_gas_giant = None
        has_rings = None

        if isinstance(md, dict):
            ss = md.get("system_summary", {})
            if isinstance(ss, dict):
                body_count = ss.get("body_count")
                stars_count = ss.get("stars_count")
                planets_count = ss.get("planets_count")

            # Infer gas giants / rings from bodies list if present
            bodies = md.get("bodies")
            if isinstance(bodies, list):
                gg = 0
                rings = 0
                for b in bodies:
                    if not isinstance(b, dict):
                        continue
                    pc = b.get("planet_class") or b.get("PlanetClass")
                    if isinstance(pc, str):
                        pcl = pc.lower()
                        if "gas giant" in pcl or "sudarsky" in pcl or "helium" in pcl:
                            gg += 1
                    rs = b.get("rings") or b.get("Rings")
                    if isinstance(rs, list) and len(rs) > 0:
                        rings += 1
                has_gas_giant = (gg > 0) if bodies else None
                has_rings = (rings > 0) if bodies else None

        rr = dict(r)
        rr.update({
            "body_count": body_count,
            "stars_count": stars_count,
            "planets_count": planets_count,
            "has_gas_giant": has_gas_giant,
            "has_rings": has_rings,
            "has_metadata": md is not None,
        })
        out.append(rr)
    return out


def load_nonclick() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not NONCLICK_DIR.exists():
        return rows
    for fp in NONCLICK_DIR.glob("*.json"):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        sys = d.get("system", {})
        feats = d.get("features", {})
        stars = feats.get("stars", {}) or {}
        bodies = feats.get("bodies", {}) or {}
        pos = _safe_pos(sys.get("star_pos"))
        rows.append({
            "label": "non_click",
            "name": sys.get("name"),
            "address": sys.get("address"),
            "pos": pos,
            "body_count": bodies.get("fss_bodycount"),
            "stars_count": stars.get("n_stars"),
            "planets_count": bodies.get("n_planets_scanned"),
            "has_gas_giant": bodies.get("has_gas_giant"),
            "has_rings": bodies.get("has_ringed_body"),
        })
    return rows


def save_fig(filename: str) -> Path:
    p = OUT_DIR / filename
    plt.tight_layout()
    plt.savefig(p, dpi=160)
    plt.close()
    return p


def plot_scatter_xz(click: List[Dict[str, Any]], non: List[Dict[str, Any]]) -> Path:
    plt.figure()
    cx = [r["pos"][0] for r in click if r.get("pos")]
    cz = [r["pos"][2] for r in click if r.get("pos")]
    nx = [r["pos"][0] for r in non if r.get("pos")]
    nz = [r["pos"][2] for r in non if r.get("pos")]
    plt.scatter(cx, cz, alpha=0.7, label="click")
    plt.scatter(nx, nz, alpha=0.35, label="non_click")
    plt.title("Galactic X-Z scatter")
    plt.xlabel("X (ly)")
    plt.ylabel("Z (ly)")
    plt.legend()
    return save_fig("click_vs_nonclick_scatter_xz.png")


def plot_hist_dist_to_sol(click: List[Dict[str, Any]], non: List[Dict[str, Any]]) -> Path:
    sol = (0.0, 0.0, 0.0)
    cd = [_dist3(r["pos"], sol) for r in click if r.get("pos")]
    nd = [_dist3(r["pos"], sol) for r in non if r.get("pos")]
    plt.figure()
    if cd:
        plt.hist(cd, bins=30, alpha=0.5, label="click")
    if nd:
        plt.hist(nd, bins=30, alpha=0.5, label="non_click")
    plt.title("Distance to Sol")
    plt.xlabel("Distance (ly)")
    plt.ylabel("Count")
    plt.legend()
    return save_fig("click_vs_nonclick_hist_dist_to_sol.png")


def plot_hist_absY(click: List[Dict[str, Any]], non: List[Dict[str, Any]]) -> Path:
    cy = [abs(r["pos"][1]) for r in click if r.get("pos")]
    ny = [abs(r["pos"][1]) for r in non if r.get("pos")]
    plt.figure()
    if cy:
        plt.hist(cy, bins=30, alpha=0.5, label="click")
    if ny:
        plt.hist(ny, bins=30, alpha=0.5, label="non_click")
    plt.title("|Y| distance to galactic plane")
    plt.xlabel("|Y| (ly)")
    plt.ylabel("Count")
    plt.legend()
    return save_fig("click_vs_nonclick_hist_absY.png")


def plot_hist(click: List[Dict[str, Any]], non: List[Dict[str, Any]], key: str, title: str, fname: str) -> Optional[Path]:
    cv = [r.get(key) for r in click if isinstance(r.get(key), (int, float))]
    nv = [r.get(key) for r in non if isinstance(r.get(key), (int, float))]
    if not cv and not nv:
        return None
    plt.figure()
    if cv:
        plt.hist(cv, bins=30, alpha=0.5, label="click")
    if nv:
        plt.hist(nv, bins=30, alpha=0.5, label="non_click")
    plt.title(title)
    plt.xlabel(key)
    plt.ylabel("Count")
    plt.legend()
    return save_fig(fname)


def plot_ratio_bar(click: List[Dict[str, Any]], non: List[Dict[str, Any]], key: str, title: str, fname: str) -> Optional[Path]:
    def ratio(rows: List[Dict[str, Any]]) -> Optional[float]:
        vals = [r.get(key) for r in rows if isinstance(r.get(key), bool)]
        if not vals:
            return None
        return sum(1 for v in vals if v) / len(vals)

    rc = ratio(click)
    rn = ratio(non)
    if rc is None and rn is None:
        return None

    labels = []
    values = []
    if rc is not None:
        labels.append("click")
        values.append(rc)
    if rn is not None:
        labels.append("non_click")
        values.append(rn)

    plt.figure()
    plt.bar(labels, values)
    plt.ylim(0, 1)
    plt.title(title)
    plt.ylabel("ratio")
    return save_fig(fname)


def main() -> int:
    if not CLICK_SUMMARY.exists():
        raise SystemExit(f"Missing {CLICK_SUMMARY}")
    if not NONCLICK_DIR.exists():
        raise SystemExit(f"Missing {NONCLICK_DIR} (run extract_non_click_systems.py first)")

    click_base = load_click_from_summary()
    click = enrich_click_with_metadata(click_base)
    non = load_nonclick()

    paths: List[Path] = []
    paths.append(plot_scatter_xz(click, non))
    paths.append(plot_hist_dist_to_sol(click, non))
    paths.append(plot_hist_absY(click, non))

    # Enriched comparisons
    p = plot_hist(click, non, "body_count", "Body count (click: metadata, non_click: FSS bodycount)", "click_vs_nonclick_hist_bodycount.png")
    if p: paths.append(p)
    p = plot_hist(click, non, "stars_count", "Stars count", "click_vs_nonclick_hist_stars_count.png")
    if p: paths.append(p)
    p = plot_hist(click, non, "planets_count", "Planets count", "click_vs_nonclick_hist_planets_count.png")
    if p: paths.append(p)
    p = plot_ratio_bar(click, non, "has_gas_giant", "Share of systems with gas giants", "click_vs_nonclick_gas_giant_ratio.png")
    if p: paths.append(p)
    p = plot_ratio_bar(click, non, "has_rings", "Share of systems with rings", "click_vs_nonclick_rings_ratio.png")
    if p: paths.append(p)

    # Console summary
    meta_ok = sum(1 for r in click if r.get("has_metadata"))
    print(f"CLICK systems (summary.json): {len(click_base)}")
    print(f"CLICK systems enriched via metadata: {meta_ok}/{len(click)}")
    print(f"NON-CLICK systems (systems/non_click): {len(non)}")
    print("Wrote figures:")
    for p in paths:
        print(" -", p.as_posix())

    if meta_ok == 0:
        print("\nNote: No metadata matched click system names. Ensure metadata files are in ./metadata and include raw events with StarSystem/SystemName.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
