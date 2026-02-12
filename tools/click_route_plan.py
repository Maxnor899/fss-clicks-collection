#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
click_route_plan.py

Génère un plan de route (FRONTIÈRES + SPOKES) pour rechercher de nouveaux "systèmes à clics"
dans Elite Dangerous.

Spécification (figée avec Max):
- Positifs P = liste canonique des systèmes à clics (repo/ingestion). Ces systèmes sont exclus des cibles.
- Négatifs N = tous les systèmes déjà visités dans les logs Journal.*.log qui ne sont pas dans P. Exclus aussi.
- Départ = dernier système connu dans les logs (nom affiché dans le plan).
- Deux approches, chacune en UN seul chemin:
  - FRONTIÈRES (15 systèmes) : autour des clusters (DBSCAN) des positifs, allocation proportionnelle par cluster.
  - SPOKES (15 systèmes) : le long des axes principaux (PCA) depuis le barycentre global.
- Résolution OBLIGATOIRE des points (x,y,z) -> noms de systèmes jumpables :
  - tentative Spansh puis EDSM (fallback), sinon EDSM directement (Spansh peut être instable/non documenté)
  - sélection du candidat le plus proche du point
  - on n'accepte que les noms ∉ P, ∉ N, et pas déjà pris dans la route.
- Sortie unique: route_plan.md (séquences de noms uniquement, pas de distances).

Dépendances:
- numpy, scikit-learn, requests

EDSM sphere-systems (doc):
GET https://www.edsm.net/api-v1/sphere-systems?x=..&y=..&z=..&radius=..&showCoordinates=1
(voir docs API Systems v1) https://www.edsm.net/en/api-v1  (sphere-systems, radius max 100)

Usage:
  python click_route_plan.py \
    --logs "C:/Users/<you>/Saved Games/Frontier Developments/Elite Dangerous" \
    --clicks path/to/click_systems.json \
    --out route_plan.md

Formats supportés pour --clicks:
- JSON liste de noms: ["Sys A", "Sys B", ...]
- JSON liste d'objets: [{"system":"Sys A","coords":[x,y,z]}, ...] (clés acceptées: system/name/StarSystem ; coords/StarPos/starpos)
- JSON dict: {"Sys A": {...meta...}, "Sys B": {...}} (coords dans la meta)
- TXT: un nom de système par ligne

"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import requests
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# -------------------------
# Data models
# -------------------------

@dataclass(frozen=True)
class SystemPoint:
    name: str
    xyz: Tuple[float, float, float]


@dataclass(frozen=True)
class ResolvedSystem:
    name: str
    xyz: Optional[Tuple[float, float, float]]  # may be None if resolver didn't return coords


# -------------------------
# Utilities
# -------------------------

def parse_iso_z(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def euclid(a: Sequence[float], b: Sequence[float]) -> float:
    return float(math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2))


def safe_float_triplet(v: Any) -> Optional[Tuple[float, float, float]]:
    if isinstance(v, (list, tuple)) and len(v) == 3:
        try:
            return (float(v[0]), float(v[1]), float(v[2]))
        except Exception:
            return None
    if isinstance(v, dict) and {"x", "y", "z"}.issubset(v.keys()):
        try:
            return (float(v["x"]), float(v["y"]), float(v["z"]))
        except Exception:
            return None
    return None


# -------------------------
# Loading click systems (positifs)
# -------------------------

def load_clicks(path: Path) -> List[SystemPoint]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".txt":
        names = []
        for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                names.append(ln)
        return [SystemPoint(n, (math.nan, math.nan, math.nan)) for n in names]

    data = json.loads(path.read_text(encoding="utf-8"))
    out: List[SystemPoint] = []

    def coords_from_meta(meta: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
        for key in ("coords", "StarPos", "starpos", "position", "pos"):
            if key in meta:
                c = safe_float_triplet(meta[key])
                if c:
                    return c
        # sometimes nested, e.g. {"coords": {"x":..}}
        return None

    if isinstance(data, list):
        # ["name", ...] OR [{"system":..,"coords":[..]}, ...]
        for item in data:
            if isinstance(item, str):
                out.append(SystemPoint(item.strip(), (math.nan, math.nan, math.nan)))
            elif isinstance(item, dict):
                name = item.get("system") or item.get("name") or item.get("StarSystem")
                if not name:
                    continue
                c = coords_from_meta(item) or safe_float_triplet(item.get("coords")) or safe_float_triplet(item.get("StarPos"))
                out.append(SystemPoint(str(name), c if c else (math.nan, math.nan, math.nan)))
    elif isinstance(data, dict):
        # {"Name": meta, ...} OR {"systems":[...]}
        if "systems" in data and isinstance(data["systems"], list):
            for item in data["systems"]:
                if isinstance(item, str):
                    out.append(SystemPoint(item.strip(), (math.nan, math.nan, math.nan)))
                elif isinstance(item, dict):
                    name = item.get("system") or item.get("name") or item.get("StarSystem")
                    if not name:
                        continue
                    c = coords_from_meta(item)
                    out.append(SystemPoint(str(name), c if c else (math.nan, math.nan, math.nan)))
        else:
            for name, meta in data.items():
                if isinstance(meta, dict):
                    c = coords_from_meta(meta)
                    out.append(SystemPoint(str(name), c if c else (math.nan, math.nan, math.nan)))
                else:
                    out.append(SystemPoint(str(name), (math.nan, math.nan, math.nan)))
    else:
        raise ValueError("Format clicks non supporté")

    # sanity: drop empties
    out = [sp for sp in out if sp.name]
    return out


# -------------------------
# Parsing journal logs (visited + last system)
# -------------------------

def iter_journal_files(log_dir: Path) -> List[Path]:
    files = sorted(log_dir.glob("Journal.*.log"))
    return [p for p in files if p.is_file()]


def iter_events(files: List[Path]) -> Iterable[Dict[str, Any]]:
    for fp in files:
        with fp.open("r", encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    yield json.loads(ln)
                except Exception:
                    continue


def extract_visited_and_last(log_dir: Path) -> Tuple[Set[str], str, Optional[Tuple[float, float, float]]]:
    files = iter_journal_files(log_dir)
    if not files:
        raise FileNotFoundError(f"Aucun Journal.*.log dans {log_dir}")

    visited: Set[str] = set()
    last_name: Optional[str] = None
    last_ts: Optional[datetime] = None
    last_xyz: Optional[Tuple[float, float, float]] = None

    for ev in iter_events(files):
        et = ev.get("event")
        if et not in ("Location", "FSDJump", "CarrierJump"):
            continue
        ts = parse_iso_z(ev.get("timestamp", ""))
        name = ev.get("StarSystem")
        if not name or not ts:
            continue
        visited.add(str(name))
        if (last_ts is None) or (ts >= last_ts):
            last_ts = ts
            last_name = str(name)
            sp = safe_float_triplet(ev.get("StarPos"))
            last_xyz = sp if sp else last_xyz

    if not last_name:
        raise RuntimeError("Impossible de déterminer le dernier système depuis les logs.")
    return visited, last_name, last_xyz


# -------------------------
# Geometry: DBSCAN clusters + PCA axes
# -------------------------

def coords_matrix(positives: List[SystemPoint]) -> Tuple[np.ndarray, List[str]]:
    names = []
    pts = []
    for sp in positives:
        x, y, z = sp.xyz
        if any(math.isnan(v) for v in (x, y, z)):
            continue
        names.append(sp.name)
        pts.append([x, y, z])
    if not pts:
        return np.empty((0, 3)), []
    return np.array(pts, dtype=float), names


def median_nn_distance(X: np.ndarray) -> float:
    if X.shape[0] < 2:
        return 20.0
    nn = NearestNeighbors(n_neighbors=2).fit(X)
    d, _ = nn.kneighbors(X)
    return float(np.median(d[:, 1]))


def cluster_dbscan(X: np.ndarray) -> np.ndarray:
    # eps based on median NN
    med = median_nn_distance(X)
    eps = max(1.0, med * 1.5)
    labels = DBSCAN(eps=eps, min_samples=3).fit_predict(X)
    return labels


def pca_axes(X: np.ndarray) -> np.ndarray:
    # returns 3x3 components
    pca = PCA(n_components=3)
    pca.fit(X)
    return pca.components_


# -------------------------
# Target point generation
# -------------------------

def golden_sphere_points(n: int) -> List[Tuple[float, float, float]]:
    pts = []
    n = max(4, int(n))
    for k in range(n):
        i = k + 0.5
        phi = math.acos(1 - 2 * i / n)
        theta = math.pi * (1 + 5 ** 0.5) * i
        x = math.sin(phi) * math.cos(theta)
        y = math.sin(phi) * math.sin(theta)
        z = math.cos(phi)
        pts.append((x, y, z))
    return pts


def make_frontier_points_per_cluster(X: np.ndarray, labels: np.ndarray, target_total: int) -> List[Tuple[float, float, float]]:
    """
    Allocation proportionnelle aux clusters (label != -1).
    Chaque cluster génère des points sur une coquille autour de son barycentre.
    """
    frontier: List[Tuple[float, float, float]] = []
    cluster_ids = [c for c in sorted(set(labels.tolist())) if c != -1]
    if not cluster_ids:
        # fallback: treat all as one cluster
        cluster_ids = [0]
        labels = np.zeros((X.shape[0],), dtype=int)

    # sizes
    sizes = {c: int(np.sum(labels == c)) for c in cluster_ids}
    total = sum(sizes.values())
    # provisional allocation
    alloc = {c: max(1, round(target_total * sizes[c] / total)) for c in cluster_ids}
    # adjust to exact target_total
    while sum(alloc.values()) > target_total:
        # remove from largest allocated
        c = max(alloc.keys(), key=lambda k: alloc[k])
        if alloc[c] > 1:
            alloc[c] -= 1
        else:
            break
    while sum(alloc.values()) < target_total:
        c = max(sizes.keys(), key=lambda k: sizes[k])
        alloc[c] += 1

    for c in cluster_ids:
        Xi = X[labels == c]
        if Xi.shape[0] == 0:
            continue
        bary = Xi.mean(axis=0)
        r = max(5.0, median_nn_distance(Xi) * 1.8)
        dirs = golden_sphere_points(alloc[c])
        for dx, dy, dz in dirs:
            p = bary + r * np.array([dx, dy, dz])
            frontier.append((float(p[0]), float(p[1]), float(p[2])))

    return frontier


def make_spoke_points(X: np.ndarray, target_total: int) -> List[Tuple[float, float, float]]:
    """
    Génère des points sur des rayons depuis le barycentre global, orientés selon PCA.
    """
    if X.shape[0] == 0:
        return []
    bary = X.mean(axis=0)
    axes = pca_axes(X)  # 3 axes
    step = max(5.0, median_nn_distance(X) * 1.8)

    # We'll generate plenty, then cut to target_total
    points: List[Tuple[float, float, float]] = []
    steps_per_dir = max(2, math.ceil(target_total / 6))
    for ax in axes[:3]:
        ax = ax / np.linalg.norm(ax)
        for sign in (+1, -1):
            for s in range(1, steps_per_dir + 3):
                p = bary + (sign * ax) * step * s
                points.append((float(p[0]), float(p[1]), float(p[2])))

    # keep first target_total (order doesn't matter now; route optimizer will)
    return points[:target_total]


# -------------------------
# Resolver: Spansh (best-effort) then EDSM (reliable, documented)
# -------------------------

class Resolver:
    def __init__(self, prefer: str = "spansh_then_edsm", user_agent: str = "click-route-plan/1.0"):
        self.prefer = prefer
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def resolve_nearest(self, point: Tuple[float, float, float], exclude: Set[str], taken: Set[str]) -> Optional[ResolvedSystem]:
        """
        Resolve a point to a real system name. Must return a name not in exclude and not in taken.
        """
        x, y, z = point
        if self.prefer == "edsm_only":
            return self._resolve_via_edsm(point, exclude, taken)
        # default: spansh then edsm
        rs = self._resolve_via_spansh(point, exclude, taken)
        if rs:
            return rs
        return self._resolve_via_edsm(point, exclude, taken)

    def _resolve_via_spansh(self, point: Tuple[float, float, float], exclude: Set[str], taken: Set[str]) -> Optional[ResolvedSystem]:
        """
        Spansh has a 'nearest' feature on the website, but its API isn't consistently documented.
        We try a couple of common patterns; on failure, return None (EDSM fallback will handle).
        """
        x, y, z = point
        candidates: List[ResolvedSystem] = []

        # Attempt 1: /api/nearest?x=..&y=..&z=..
        # (best-effort guess; if it 404s, we just fallback)
        urls = [
            ("https://www.spansh.co.uk/api/nearest", {"x": x, "y": y, "z": z}),
            ("https://spansh.co.uk/api/nearest", {"x": x, "y": y, "z": z}),
        ]
        for url, params in urls:
            try:
                r = self.session.get(url, params=params, timeout=10)
                if r.status_code != 200:
                    continue
                data = r.json()
                # try to interpret
                # common shapes: {"name":"Sol","x":0,...} or {"system":{"name":..,"coords":..}}
                name = None
                xyz = None
                if isinstance(data, dict):
                    if "name" in data and isinstance(data["name"], str):
                        name = data["name"]
                        xyz = safe_float_triplet(data.get("coords")) or safe_float_triplet({"x": data.get("x"), "y": data.get("y"), "z": data.get("z")})
                    elif "system" in data and isinstance(data["system"], dict):
                        sd = data["system"]
                        name = sd.get("name")
                        xyz = safe_float_triplet(sd.get("coords")) or safe_float_triplet(sd.get("StarPos"))
                if name and (name not in exclude) and (name not in taken):
                    return ResolvedSystem(name=name, xyz=xyz)
            except Exception:
                continue

        return None

    def _resolve_via_edsm(self, point: Tuple[float, float, float], exclude: Set[str], taken: Set[str]) -> Optional[ResolvedSystem]:
        """
        Use EDSM sphere-systems endpoint. radius max 100; we expand progressively and pick
        the candidate closest to the point.
        """
        x, y, z = point
        radii = [5, 10, 20, 40, 80, 100]
        base_url = "https://www.edsm.net/api-v1/sphere-systems"

        best: Optional[ResolvedSystem] = None
        best_d = float("inf")

        for radius in radii:
            try:
                params = {
                    "x": x, "y": y, "z": z,
                    "radius": radius,
                    "minRadius": 0,
                    "showCoordinates": 1,
                }
                r = self.session.get(base_url, params=params, timeout=15)
                if r.status_code != 200:
                    continue
                data = r.json()
                if not isinstance(data, list) or len(data) == 0:
                    continue

                for item in data:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    if not name or not isinstance(name, str):
                        continue
                    if name in exclude or name in taken:
                        continue
                    coords = None
                    if "coords" in item and isinstance(item["coords"], dict):
                        coords = safe_float_triplet(item["coords"])
                    d = item.get("distance")
                    if isinstance(d, (int, float)):
                        # distance from center of sphere, not from the exact point? Actually sphere center IS point => ok.
                        dd = float(d)
                    elif coords:
                        dd = euclid(point, coords)
                    else:
                        continue
                    if dd < best_d:
                        best_d = dd
                        best = ResolvedSystem(name=name, xyz=coords)

                # if we found something at a small radius, it's likely already closest
                if best and radius <= 20:
                    return best

            except Exception:
                continue

        return best


# -------------------------
# Route optimization (nearest-neighbor)
# -------------------------

def order_nearest_neighbor(start_xyz: Optional[Tuple[float, float, float]], systems: List[ResolvedSystem]) -> List[ResolvedSystem]:
    if not systems:
        return []
    if start_xyz is None:
        # keep as-is
        return systems

    remaining = systems[:]
    ordered: List[ResolvedSystem] = []
    current = start_xyz

    while remaining:
        # choose closest among those with coords; if coords missing, push them to the end
        best_i = None
        best_d = float("inf")
        for i, s in enumerate(remaining):
            if s.xyz is None:
                continue
            d = euclid(current, s.xyz)
            if d < best_d:
                best_d = d
                best_i = i

        if best_i is None:
            # no coords available for remaining
            ordered.extend(remaining)
            break

        chosen = remaining.pop(best_i)
        ordered.append(chosen)
        if chosen.xyz is not None:
            current = chosen.xyz

    return ordered


# -------------------------
# Main generation pipeline
# -------------------------

def generate_route(
    points: List[Tuple[float, float, float]],
    resolver: Resolver,
    exclude: Set[str],
    taken_global: Set[str],
    count: int,
    seed: int = 0,
) -> List[ResolvedSystem]:
    """
    Resolve points into unique systems. If some points can't be resolved, we generate
    additional jittered points around the originals until we reach count (best-effort).
    """
    random.seed(seed)
    resolved: List[ResolvedSystem] = []
    taken_local: Set[str] = set()

    # helper to try resolve a point with a few jitters
    def try_point(p: Tuple[float, float, float]) -> Optional[ResolvedSystem]:
        # 1) direct
        rs = resolver.resolve_nearest(p, exclude, taken_global | taken_local)
        if rs:
            return rs
        # 2) jitter attempts (small offsets)
        for _ in range(6):
            j = (random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(-3, 3))
            pj = (p[0] + j[0], p[1] + j[1], p[2] + j[2])
            rs = resolver.resolve_nearest(pj, exclude, taken_global | taken_local)
            if rs:
                return rs
        return None

    # iterate points, collect
    for p in points:
        if len(resolved) >= count:
            break
        rs = try_point(p)
        if not rs:
            continue
        taken_local.add(rs.name)
        resolved.append(rs)

    # if missing, generate extra random sphere points around barycentre of the original points
    if len(resolved) < count and points:
        arr = np.array(points, dtype=float)
        bary = arr.mean(axis=0)
        base_r = float(np.median(np.linalg.norm(arr - bary, axis=1))) if arr.shape[0] > 0 else 30.0
        for k in range(200):
            if len(resolved) >= count:
                break
            dx, dy, dz = random.choice(golden_sphere_points(32))
            r = base_r * random.uniform(0.7, 1.3)
            p = (float(bary[0] + dx * r), float(bary[1] + dy * r), float(bary[2] + dz * r))
            rs = try_point(p)
            if not rs:
                continue
            taken_local.add(rs.name)
            resolved.append(rs)

    # register in global taken
    for rs in resolved:
        taken_global.add(rs.name)

    return resolved


def write_route_plan(
    out_path: Path,
    start_system: str,
    front: List[ResolvedSystem],
    spokes: List[ResolvedSystem],
):
    def chain(start: str, lst: List[ResolvedSystem]) -> str:
        names = [start] + [s.name for s in lst]
        return " -> ".join(names)

    lines = []
    lines.append(f"Start system: {start_system}")
    lines.append("")
    lines.append("FRONTIÈRES (15):")
    lines.append(chain(start_system, front))
    lines.append("")
    lines.append("SPOKES (15):")
    lines.append(chain(start_system, spokes))
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# -------------------------
# CLI
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, type=Path, help="Dossier contenant les Journal.*.log")
    ap.add_argument("--clicks", required=True, type=Path, help="Liste des systèmes à clics (json/txt) avec coords si possible")
    ap.add_argument("--out", required=True, type=Path, help="Fichier route_plan.md à produire")
    ap.add_argument("--resolver", choices=["spansh_then_edsm", "edsm_only"], default="spansh_then_edsm",
                    help="Resolver à utiliser (Spansh best-effort puis EDSM, ou EDSM seul)")
    ap.add_argument("--seed", type=int, default=0, help="Seed pour la génération (reproductibilité)")
    args = ap.parse_args()

    positives = load_clicks(args.clicks)
    P_names = {sp.name for sp in positives}
    X, names_with_coords = coords_matrix(positives)
    if X.shape[0] < 3:
        raise RuntimeError("Pas assez de coords pour les positifs. Il faut au moins 3 systèmes à clics avec coords.")

    visited, last_system, last_xyz = extract_visited_and_last(args.logs)
    # Négatifs = visités - positifs ; mais on exclut tout 'visited' au final (positifs inclus aussi)
    exclude = set(visited) | set(P_names)

    labels = cluster_dbscan(X)

    # Generate target points (raw coords)
    frontier_points = make_frontier_points_per_cluster(X, labels, target_total=15)
    spoke_points = make_spoke_points(X, target_total=15)

    resolver = Resolver(prefer=args.resolver)
    taken_global: Set[str] = set()

    frontier_systems = generate_route(frontier_points, resolver, exclude=exclude, taken_global=taken_global, count=15, seed=args.seed + 1)
    spoke_systems = generate_route(spoke_points, resolver, exclude=exclude, taken_global=taken_global, count=15, seed=args.seed + 2)

    # Order each route from start (nearest-neighbor) if coords available
    frontier_systems = order_nearest_neighbor(last_xyz, frontier_systems)
    spoke_systems = order_nearest_neighbor(last_xyz, spoke_systems)

    # Ensure length 15 (if not enough found, we still write whatever was found)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_route_plan(args.out, last_system, frontier_systems[:15], spoke_systems[:15])

    print(f"Wrote: {args.out}")
    print(f"Start system: {last_system}")
    print(f"Frontières: {len(frontier_systems[:15])}/15  | Spokes: {len(spoke_systems[:15])}/15")


if __name__ == "__main__":
    main()
