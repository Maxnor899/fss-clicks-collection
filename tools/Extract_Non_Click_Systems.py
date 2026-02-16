#!/usr/bin/env python3
"""Extract visited systems from Elite Dangerous Journal logs and write one JSON per NON-click system.

This creates a version-controlled dataset for future correlations:
- CLICK systems already exist in analysis/summary.json (your repo's truth source).
- This script builds NON-CLICK systems from Journal.*.log files, enriched with scan-derived features.

References:
- Elite Dangerous Player Journal (FSDJump/Location for SystemAddress + StarPos, Scan for bodies).
  Official journal manual PDF and community mirrors describe fields like StarSystem, SystemAddress, StarPos, StarType, PlanetClass, Rings, TerraformState.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def parse_utc(ts: str) -> Optional[datetime]:
    try:
        if ts.endswith("Z"):
            ts = ts[:-1] + "+00:00"
        return datetime.fromisoformat(ts).astimezone(timezone.utc)
    except Exception:
        return None


def safe_slug(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^\w\-\.\(\)\s]", "_", name, flags=re.UNICODE)
    name = re.sub(r"\s+", "_", name)
    return name[:160] if len(name) > 160 else name


def iter_journal_events(journal_dir: Path) -> Iterable[tuple[Path, Dict[str, Any]]]:
    for fp in sorted(journal_dir.glob("Journal.*.log")):
        try:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(ev, dict) and "event" in ev:
                        yield fp, ev
        except OSError:
            continue


def load_click_addresses(clicks_summary: Optional[Path]) -> Set[int]:
    if not clicks_summary or not clicks_summary.exists():
        return set()
    try:
        data = json.loads(clicks_summary.read_text(encoding="utf-8"))
        out: Set[int] = set()
        for s in data.get("systems", []):
            addr = s.get("system_address") or s.get("SystemAddress")
            if isinstance(addr, int):
                out.add(addr)
        return out
    except Exception:
        return set()


def is_gas_giant_planetclass(planet_class: str) -> bool:
    pc = (planet_class or "").lower()
    return ("gas giant" in pc) or ("sudarsky" in pc) or ("helium" in pc)


@dataclass
class SystemAgg:
    name: str
    address: int
    star_pos: Optional[List[float]] = None

    first_seen_utc: Optional[str] = None
    last_seen_utc: Optional[str] = None
    journal_files: Set[str] = field(default_factory=set)

    fss_bodycount: Optional[int] = None
    fss_nonbodycount: Optional[int] = None

    body_names: Set[str] = field(default_factory=set)
    star_types: Set[str] = field(default_factory=set)
    planet_classes: Set[str] = field(default_factory=set)

    n_stars: int = 0
    n_bodies_scanned: int = 0
    n_planets_scanned: int = 0

    has_gas_giant: bool = False
    n_gas_giants: int = 0

    has_ringed_body: bool = False
    n_ringed_bodies: int = 0

    n_terraformables: int = 0

    n_saa_mappings: int = 0
    n_fss_body_signals_events: int = 0

    def touch(self, ts: Optional[datetime], journal_file: str) -> None:
        self.journal_files.add(journal_file)
        if ts is None:
            return
        iso = ts.isoformat().replace("+00:00", "Z")
        if self.first_seen_utc is None or iso < self.first_seen_utc:
            self.first_seen_utc = iso
        if self.last_seen_utc is None or iso > self.last_seen_utc:
            self.last_seen_utc = iso

    def to_json(self, label: str, generated_by: str) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "label": label,
            "system": {
                "name": self.name,
                "address": self.address,
                "star_pos": self.star_pos,
            },
            "evidence": {
                "method": "elite_journal",
                "journal": {
                    "first_seen_utc": self.first_seen_utc,
                    "last_seen_utc": self.last_seen_utc,
                    "files": sorted(self.journal_files),
                },
            },
            "features": {
                "stars": {
                    "n_stars": self.n_stars or None,
                    "types": sorted(self.star_types),
                },
                "bodies": {
                    "n_bodies_scanned": self.n_bodies_scanned or None,
                    "n_planets_scanned": self.n_planets_scanned or None,
                    "fss_bodycount": self.fss_bodycount,
                    "fss_nonbodycount": self.fss_nonbodycount,
                    "has_gas_giant": self.has_gas_giant,
                    "n_gas_giants": self.n_gas_giants,
                    "has_ringed_body": self.has_ringed_body,
                    "n_ringed_bodies": self.n_ringed_bodies,
                    "n_terraformables": self.n_terraformables,
                    "planet_classes": sorted(self.planet_classes),
                },
                "signals": {
                    "n_saa_mappings": self.n_saa_mappings,
                    "n_fss_body_signals_events": self.n_fss_body_signals_events,
                },
            },
            "provenance": {
                "generated_by": generated_by,
                "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            },
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal-dir", required=True, help="Elite Dangerous journal dir containing Journal.*.log")
    ap.add_argument("--clicks-summary", default="analysis/summary.json", help="Path to analysis/summary.json (click systems)")
    ap.add_argument("--out-dir", default="systems/non_click", help="Output directory for non-click JSONs")
    ap.add_argument("--include-click", action="store_true", help="Also write click systems to systems/click/")
    ap.add_argument("--click-out-dir", default="systems/click", help="Output directory for click systems if --include-click")
    args = ap.parse_args()

    journal_dir = Path(args.journal_dir).expanduser()
    clicks_summary = Path(args.clicks_summary).expanduser() if args.clicks_summary else None
    out_dir = Path(args.out_dir).expanduser()
    click_out_dir = Path(args.click_out_dir).expanduser()

    click_addrs = load_click_addresses(clicks_summary)

    systems: Dict[int, SystemAgg] = {}

    def get_sys(addr: int, name: str) -> SystemAgg:
        if addr not in systems:
            systems[addr] = SystemAgg(name=name, address=addr)
        else:
            if name and systems[addr].name != name:
                systems[addr].name = name
        return systems[addr]

    for fp, ev in iter_journal_events(journal_dir):
        ts = parse_utc(str(ev.get("timestamp", "")))
        etype = ev.get("event")
        jfile = fp.name

        if etype in ("FSDJump", "Location"):
            name = ev.get("StarSystem")
            addr = ev.get("SystemAddress")
            pos = ev.get("StarPos")
            if isinstance(addr, int) and isinstance(name, str) and name:
                sys = get_sys(addr, name)
                if isinstance(pos, list) and len(pos) == 3:
                    sys.star_pos = [float(pos[0]), float(pos[1]), float(pos[2])]
                sys.touch(ts, jfile)

        if etype == "FSSDiscoveryScan":
            addr = ev.get("SystemAddress")
            if isinstance(addr, int):
                name = ev.get("SystemName") or ""
                sys = get_sys(addr, name if isinstance(name, str) else "")
                bc = ev.get("BodyCount")
                nbc = ev.get("NonBodyCount")
                if isinstance(bc, int):
                    sys.fss_bodycount = bc
                if isinstance(nbc, int):
                    sys.fss_nonbodycount = nbc
                sys.touch(ts, jfile)

        if etype == "Scan":
            addr = ev.get("SystemAddress")
            if not isinstance(addr, int):
                continue
            name = ev.get("StarSystem") or ev.get("SystemName") or ""
            sys = get_sys(addr, name if isinstance(name, str) else "")
            sys.touch(ts, jfile)

            bname = ev.get("BodyName")
            if isinstance(bname, str) and bname:
                sys.body_names.add(bname)

            st = ev.get("StarType")
            pc = ev.get("PlanetClass")

            if isinstance(st, str) and st:
                sys.star_types.add(st)
                sys.n_stars = max(sys.n_stars, len(sys.star_types))
                sys.n_bodies_scanned += 1
            elif isinstance(pc, str) and pc:
                sys.n_bodies_scanned += 1
                sys.n_planets_scanned += 1
                sys.planet_classes.add(pc)

                if is_gas_giant_planetclass(pc):
                    sys.has_gas_giant = True
                    sys.n_gas_giants += 1

                tf = ev.get("TerraformState")
                if isinstance(tf, str) and tf.lower().startswith("terraform"):
                    sys.n_terraformables += 1

            rings = ev.get("Rings")
            if isinstance(rings, list) and len(rings) > 0:
                sys.has_ringed_body = True
                sys.n_ringed_bodies += 1

        if etype == "SAAScanComplete":
            addr = ev.get("SystemAddress")
            if isinstance(addr, int):
                name = ev.get("SystemName") or ""
                sys = get_sys(addr, name if isinstance(name, str) else "")
                sys.n_saa_mappings += 1
                sys.touch(ts, jfile)

        if etype == "FSSBodySignals":
            addr = ev.get("SystemAddress")
            if isinstance(addr, int):
                name = ev.get("SystemName") or ""
                sys = get_sys(addr, name if isinstance(name, str) else "")
                sys.n_fss_body_signals_events += 1
                sys.touch(ts, jfile)

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.include_click:
        click_out_dir.mkdir(parents=True, exist_ok=True)

    written_non = 0
    written_click = 0

    for addr, sys in systems.items():
        label = "click" if addr in click_addrs else "non_click"
        payload = sys.to_json(label=label, generated_by="tools/extract_non_click_systems.py")
        fname = f"{safe_slug(sys.name)}__{addr}.json"

        if label == "non_click":
            (out_dir / fname).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            written_non += 1
        elif args.include_click:
            (click_out_dir / fname).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            written_click += 1

    print(f"Visited systems found: {len(systems)}")
    print(f"Written non-click: {written_non} -> {out_dir}")
    if args.include_click:
        print(f"Written click: {written_click} -> {click_out_dir}")
    else:
        print("Click systems detected via clicks summary, not written (use --include-click if you want them).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
