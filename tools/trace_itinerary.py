#!/usr/bin/env python3
"""
tools/trace_itinerary.py

Reconstruct Elite Dangerous itinerary from Journal logs and mark "click systems"
based on the repo's analysis artifacts (default: analysis/summary.json).

Outputs:
  - analysis/itinerary_jumps.csv
  - analysis/itinerary_segments.json
  - analysis/itinerary_jumps.json (optional, useful for plotting)

Usage examples:
  python tools/trace_itinerary.py --journals-dir SavedGames/Frontier\ Developments/Elite\ Dangerous
  python tools/trace_itinerary.py --journals-zip path/to/Journal....zip
  python tools/trace_itinerary.py --summary analysis/summary.json --out analysis

Notes:
- We detect "click systems" via analysis/summary.json (systems list).
  This is the most robust join key because it's produced by your click analysis pipeline.
- Segment detection: start a new segment when direction changes by > --angle-deg.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Data model
# ----------------------------

@dataclass
class Jump:
    timestamp: str
    system_name: str
    star_pos: Tuple[float, float, float]
    jump_dist: float
    # derived
    is_click_system: bool = False


@dataclass
class Segment:
    index: int
    start_timestamp: str
    end_timestamp: str
    start_system: str
    end_system: str
    n_jumps: int
    distance_sum_ly: float
    click_system_hits: int
    first_click_system: Optional[str]
    last_click_system: Optional[str]
    # direction summary
    mean_direction: Tuple[float, float, float]


# ----------------------------
# Helpers
# ----------------------------

def _norm(v: Tuple[float, float, float]) -> float:
    return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])


def _sub(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])


def _add(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])


def _scale(v: Tuple[float, float, float], s: float) -> Tuple[float, float, float]:
    return (v[0]*s, v[1]*s, v[2]*s)


def _unit(v: Tuple[float, float, float]) -> Tuple[float, float, float]:
    n = _norm(v)
    if n <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (v[0]/n, v[1]/n, v[2]/n)


def _dot(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def _angle_deg(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    ua = _unit(a)
    ub = _unit(b)
    d = max(-1.0, min(1.0, _dot(ua, ub)))
    return math.degrees(math.acos(d))


def _iter_journal_lines_from_dir(journals_dir: Path) -> Iterable[str]:
    # Elite journals are JSON lines; files typically named Journal.YYYY-MM-DDTHHMMSS.01.log
    for p in sorted(journals_dir.glob("Journal.*.log")):
        try:
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
        except Exception:
            continue


def _iter_journal_lines_from_zip(journals_zip: Path) -> Iterable[str]:
    with zipfile.ZipFile(journals_zip, "r") as z:
        # iterate only Journal.*.log files
        names = sorted([n for n in z.namelist() if re.search(r"Journal\..*\.log$", n)])
        for name in names:
            with z.open(name, "r") as f:
                for raw in f:
                    try:
                        line = raw.decode("utf-8", errors="ignore").strip()
                    except Exception:
                        continue
                    if line:
                        yield line


def load_click_systems_from_summary(summary_path: Path) -> set[str]:
    """
    Extract system_name from analysis/summary.json (produced by analyze_clicks.py).
    This is the cleanest "truth set" of known click systems.
    """
    with summary_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    s = set()
    for row in data.get("systems", []):
        name = row.get("system_name") or row.get("system") or row.get("name")
        if isinstance(name, str) and name.strip():
            s.add(name.strip())
    return s


def parse_fsd_jumps(lines: Iterable[str]) -> List[Jump]:
    jumps: List[Jump] = []
    for line in lines:
        try:
            ev = json.loads(line)
        except Exception:
            continue
        if ev.get("event") != "FSDJump":
            continue

        ts = ev.get("timestamp")
        system = ev.get("StarSystem")
        pos = ev.get("StarPos")
        dist = ev.get("JumpDist")

        if not (isinstance(ts, str) and isinstance(system, str) and isinstance(pos, list) and len(pos) == 3):
            continue
        if not isinstance(dist, (int, float)):
            # Some logs might omit JumpDist; we can compute it later from positions if needed,
            # but usually it's present. Skip if missing.
            continue

        try:
            star_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
            jump_dist = float(dist)
        except Exception:
            continue

        jumps.append(Jump(
            timestamp=ts,
            system_name=system,
            star_pos=star_pos,
            jump_dist=jump_dist,
        ))
    return jumps


def mark_click_systems(jumps: List[Jump], click_systems: set[str]) -> None:
    for j in jumps:
        j.is_click_system = (j.system_name in click_systems)


def detect_segments(
    jumps: List[Jump],
    angle_threshold_deg: float = 10.0,
    min_jumps_per_segment: int = 5,
) -> List[Segment]:
    """
    Segment a path into quasi-linear runs by monitoring direction changes.

    - Compute step vectors v_i = pos_i - pos_{i-1}
    - Maintain a running mean direction for current segment.
    - Start a new segment when angle(v_i, mean_dir) exceeds threshold.
    """
    if len(jumps) < 2:
        return []

    # Compute step vectors between jumps (from previous system to current system)
    step_vecs: List[Tuple[float, float, float]] = []
    for i in range(1, len(jumps)):
        v = _sub(jumps[i].star_pos, jumps[i-1].star_pos)
        step_vecs.append(v)

    segments: List[Segment] = []

    seg_start = 0  # index into jumps
    mean_dir = _unit(step_vecs[0])
    mean_dir_sum = mean_dir  # accumulate then renormalize
    mean_dir_count = 1

    def finalize(seg_end: int, mean_dir_vec: Tuple[float, float, float]) -> None:
        # seg_end is inclusive jump index
        seg_jumps = jumps[seg_start:seg_end+1]
        dist_sum = sum(j.jump_dist for j in seg_jumps[1:])  # JumpDist applies to entering system
        click_hits = sum(1 for j in seg_jumps if j.is_click_system)
        click_names = [j.system_name for j in seg_jumps if j.is_click_system]
        segments.append(Segment(
            index=len(segments),
            start_timestamp=seg_jumps[0].timestamp,
            end_timestamp=seg_jumps[-1].timestamp,
            start_system=seg_jumps[0].system_name,
            end_system=seg_jumps[-1].system_name,
            n_jumps=len(seg_jumps) - 1,
            distance_sum_ly=dist_sum,
            click_system_hits=click_hits,
            first_click_system=(click_names[0] if click_names else None),
            last_click_system=(click_names[-1] if click_names else None),
            mean_direction=_unit(mean_dir_vec),
        ))

    for i in range(1, len(step_vecs)):
        v = step_vecs[i]
        ang = _angle_deg(v, mean_dir)
        # decide split
        if ang > angle_threshold_deg:
            seg_end = i  # because step_vec i goes from jump i to i+1; last jump in segment is i
            # enforce minimum size by merging tiny segments into previous if needed
            if (seg_end - seg_start) < min_jumps_per_segment and segments:
                # merge: do nothing, keep accumulating
                pass
            else:
                finalize(seg_end, mean_dir_sum)
                seg_start = seg_end
                mean_dir = _unit(v)
                mean_dir_sum = mean_dir
                mean_dir_count = 1
                continue

        # update running mean direction
        uv = _unit(v)
        mean_dir_sum = _add(mean_dir_sum, uv)
        mean_dir_count += 1
        mean_dir = _unit(mean_dir_sum)

    # finalize last segment
    finalize(len(jumps) - 1, mean_dir_sum)
    return segments


def write_jumps_csv(path: Path, jumps: List[Jump]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "system_name", "x", "y", "z", "jump_dist_ly", "is_click_system"])
        for j in jumps:
            w.writerow([j.timestamp, j.system_name, j.star_pos[0], j.star_pos[1], j.star_pos[2], j.jump_dist, int(j.is_click_system)])


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild itinerary from Elite Journal logs and mark click systems.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--journals-dir", type=str, help="Directory containing Journal.*.log files")
    g.add_argument("--journals-zip", type=str, help="Zip containing Journal.*.log files")

    p.add_argument("--summary", type=str, default="analysis/summary.json",
                   help="analysis/summary.json (produced by analyze_clicks.py) used to identify click systems")
    p.add_argument("--out", type=str, default="analysis", help="Output directory")

    p.add_argument("--angle-deg", type=float, default=10.0,
                   help="Angle threshold (deg) to split segments")
    p.add_argument("--min-seg-jumps", type=int, default=5,
                   help="Minimum jumps per segment before allowing a split")
    p.add_argument("--dump-jumps-json", action="store_true",
                   help="Also write analysis/itinerary_jumps.json for plotting")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_dir = Path(args.out).resolve()
    summary_path = Path(args.summary).resolve()

    if not summary_path.exists():
        print(f"[ERROR] summary file not found: {summary_path}")
        print("        This script uses analysis/summary.json to identify click systems.")
        return 2

    click_systems = load_click_systems_from_summary(summary_path)
    print(f"Loaded {len(click_systems)} click systems from: {summary_path}")

    if args.journals_dir:
        lines = _iter_journal_lines_from_dir(Path(args.journals_dir).resolve())
    else:
        lines = _iter_journal_lines_from_zip(Path(args.journals_zip).resolve())

    jumps = parse_fsd_jumps(lines)
    print(f"Parsed {len(jumps)} FSDJump events")

    mark_click_systems(jumps, click_systems)
    n_hits = sum(1 for j in jumps if j.is_click_system)
    print(f"Click-system hits along itinerary: {n_hits}")

    segments = detect_segments(
        jumps,
        angle_threshold_deg=float(args.angle_deg),
        min_jumps_per_segment=int(args.min_seg_jumps),
    )
    print(f"Detected {len(segments)} segments (quasi-linear runs)")

    # Outputs
    jumps_csv = out_dir / "itinerary_jumps.csv"
    write_jumps_csv(jumps_csv, jumps)
    print(f"Wrote: {jumps_csv}")

    seg_json = out_dir / "itinerary_segments.json"
    write_json(seg_json, [asdict(s) for s in segments])
    print(f"Wrote: {seg_json}")

    if args.dump_jumps_json:
        jumps_json = out_dir / "itinerary_jumps.json"
        write_json(jumps_json, [asdict(j) for j in jumps])
        print(f"Wrote: {jumps_json}")

    # Quick console summary
    print("\nSegments summary:")
    for s in segments:
        print(
            f"  seg#{s.index:02d}  jumps={s.n_jumps:4d}  dist={s.distance_sum_ly:8.1f} ly"
            f"  clicks={s.click_system_hits:2d}"
            f"  {s.start_system} -> {s.end_system}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
