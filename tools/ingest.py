#!/usr/bin/env python3
"""
Ingest one FLAC recording into the repo.

Workflow:
- Put exactly ONE .flac file (<= 60s) into ./incoming/
- Run: python tools/ingest.py "System Name Here"
- The script will:
  - compute SHA-256 of the audio, keep first 8 chars
  - rename + move it into ./audio/<SystemName>__<hash>.flac
  - parse Elite Dangerous Journal*.log locally to extract metadata
  - write ./metadata/<SystemName>__<hash>.json
  - never copies or commits the logs

Dependencies (recommended):
  pip install mutagen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from mutagen.flac import FLAC  # type: ignore
except Exception:
    FLAC = None


# -------------------------
# Helpers
# -------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

def sanitize_system_name(name: str) -> str:
    """
    Make a filesystem-safe, git-friendly name.
    - spaces -> underscores
    - keep A-Z a-z 0-9 _ - . (and remove others)
    """
    name = name.strip().replace(" ", "_")
    name = re.sub(r"[^A-Za-z0-9_\-\.]", "", name)
    name = re.sub(r"_+", "_", name)
    return name

def get_flac_duration_seconds(path: Path) -> Optional[float]:
    """
    Return duration in seconds if possible, else None.
    Uses mutagen (pure python). If missing, advise installation.
    """
    if FLAC is None:
        return None
    audio = FLAC(str(path))
    return float(audio.info.length)

def default_elite_journal_dir() -> Path:
    """
    Default ED journal path on Windows:
    C:\\Users\\<user>\\Saved Games\\Frontier Developments\\Elite Dangerous\\
    """
    home = Path.home()
    return home / "Saved Games" / "Frontier Developments" / "Elite Dangerous"

def iter_journal_files(journal_dir: Path) -> List[Path]:
    if not journal_dir.exists():
        return []
    # Journal files are typically named Journal.<timestamp>.01.log
    files = sorted(journal_dir.glob("Journal*.log"), key=lambda p: p.stat().st_mtime)
    return files

def safe_json_loads(line: str) -> Optional[Dict[str, Any]]:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None

def hash_commander_name(commander: Optional[str]) -> Optional[str]:
    if not commander:
        return None
    return hashlib.sha256(commander.encode("utf-8")).hexdigest()[:12]


# -------------------------
# Journal extraction
# -------------------------

@dataclass
class VisitContext:
    anchor_event: Dict[str, Any]
    anchor_timestamp: str
    system_name: str
    system_address: Optional[int]
    star_pos: Optional[List[float]]
    journal_file: str

def find_latest_visit_anchor(journal_files: List[Path], target_system: str) -> Optional[VisitContext]:
    """
    Find the most recent Location/FSDJump event where StarSystem == target_system.
    """
    target_system_norm = target_system.strip()

    best: Optional[VisitContext] = None

    for jf in journal_files:
        try:
            with jf.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    obj = safe_json_loads(line)
                    if not obj:
                        continue
                    ev = obj.get("event")
                    if ev not in ("Location", "FSDJump"):
                        continue
                    if obj.get("StarSystem") != target_system_norm:
                        continue

                    ts = obj.get("timestamp")
                    if not isinstance(ts, str):
                        continue

                    ctx = VisitContext(
                        anchor_event=obj,
                        anchor_timestamp=ts,
                        system_name=obj.get("StarSystem"),
                        system_address=obj.get("SystemAddress") if isinstance(obj.get("SystemAddress"), int) else None,
                        star_pos=obj.get("StarPos") if isinstance(obj.get("StarPos"), list) else None,
                        journal_file=jf.name,
                    )
                    # Keep latest by timestamp lexicographically (ISO Z timestamps sort well)
                    if best is None or ctx.anchor_timestamp > best.anchor_timestamp:
                        best = ctx
        except Exception:
            continue

    return best

def extract_events_for_visit(
    journal_files: List[Path],
    anchor: VisitContext,
) -> Dict[str, Any]:
    """
    Collect relevant events after the anchor timestamp, while we remain in the same StarSystem.
    Stop when we detect an FSDJump to a different StarSystem (i.e., you left the system).

    Also gather the last LoadGame event before the anchor (for game version/build + commander hash).
    """
    target_system = anchor.system_name
    anchor_ts = anchor.anchor_timestamp

    # First pass: find last LoadGame before anchor
    last_loadgame: Optional[Dict[str, Any]] = None
    for jf in journal_files:
        try:
            with jf.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    obj = safe_json_loads(line)
                    if not obj:
                        continue
                    ts = obj.get("timestamp")
                    if not isinstance(ts, str):
                        continue
                    if ts >= anchor_ts:
                        # logs are chronological within file, but we don't rely on it perfectly
                        continue
                    if obj.get("event") == "LoadGame":
                        last_loadgame = obj
        except Exception:
            continue

    # Second pass: gather events in visit window
    # We'll include events if their timestamp >= anchor_ts and (where applicable) StarSystem matches.
    collected: Dict[str, List[Dict[str, Any]]] = {
        "FSSDiscoveryScan": [],
        "Scan": [],
        "SAASignalsFound": [],
        "CodexEntry": [],
        "Other": [],
    }

    visit_time_min: Optional[str] = None
    visit_time_max: Optional[str] = None
    left_system = False

    for jf in journal_files:
        try:
            with jf.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    obj = safe_json_loads(line)
                    if not obj:
                        continue
                    ts = obj.get("timestamp")
                    if not isinstance(ts, str):
                        continue
                    if ts < anchor_ts:
                        continue

                    ev = obj.get("event")

                    # Detect leaving system
                    if ev == "FSDJump":
                        ss = obj.get("StarSystem")
                        if isinstance(ss, str) and ss != target_system:
                            left_system = True
                            # update max time to just before leaving
                            if visit_time_max is None or ts > visit_time_max:
                                visit_time_max = ts
                            break

                    # Consider it part of visit if StarSystem matches when present.
                    # Many events do not carry StarSystem; in that case we accept them after anchor
                    # until we detect leaving.
                    ss = obj.get("StarSystem")
                    if isinstance(ss, str) and ss != target_system:
                        # Not our system context (e.g., other entries after jump) – ignore
                        continue

                    if visit_time_min is None:
                        visit_time_min = ts
                    visit_time_max = ts

                    if ev in collected:
                        collected[ev].append(obj)
                    else:
                        # Keep some other potentially useful events, but avoid huge dumps.
                        # You can whitelist more later if needed.
                        if ev in ("FSSSignalDiscovered", "FSSAllBodiesFound", "FSSBodySignals", "SAAEvents"):
                            collected["Other"].append(obj)
        except Exception:
            continue

        if left_system:
            break

    # Build summaries from Scan
    scans = collected["Scan"]
    stars: List[Dict[str, Any]] = []
    planets: List[Dict[str, Any]] = []

    for s in scans:
        # Stars usually have StarType; planets have PlanetClass
        if "StarType" in s:
            stars.append(prune_scan_event(s))
        elif "PlanetClass" in s:
            planets.append(prune_scan_event(s))

    # Basic counts if possible
    system_summary: Dict[str, Any] = {
        "body_count": None,
        "non_body_signals": None,
        "stars_count": len(stars) if stars else None,
        "planets_count": len(planets) if planets else None,
        "landables_count": None,
        "exobio_indicators": {
            "codex_entries_count": len(collected["CodexEntry"]) if collected["CodexEntry"] else None,
            "has_codex_biology": None,
        },
    }

    # FSSDiscoveryScan usually provides BodyCount + NonBodyCount
    if collected["FSSDiscoveryScan"]:
        last = collected["FSSDiscoveryScan"][-1]
        if isinstance(last.get("BodyCount"), int):
            system_summary["body_count"] = last.get("BodyCount")
        if isinstance(last.get("NonBodyCount"), int):
            system_summary["non_body_signals"] = last.get("NonBodyCount")

    # Landables count from Scan events
    landables = 0
    landable_seen = False
    for p in planets:
        if "Landable" in p:
            landable_seen = True
            if p.get("Landable") is True:
                landables += 1
    system_summary["landables_count"] = landables if landable_seen else None

    # Codex biology hint (best-effort; depends on event fields)
    has_bio: Optional[bool] = None
    if collected["CodexEntry"]:
        # Try to infer biology category without relying on exact strings too much
        # Some entries include "Category" or "SubCategory" etc.
        bio = False
        unknown = False
        for ce in collected["CodexEntry"]:
            cat = str(ce.get("Category", "")).lower()
            sub = str(ce.get("SubCategory", "")).lower()
            name = str(ce.get("Name", "")).lower()
            if "biology" in cat or "biology" in sub or "bio" in name:
                bio = True
            elif cat == "" and sub == "" and name == "":
                unknown = True
        if bio:
            has_bio = True
        elif unknown:
            has_bio = None
        else:
            has_bio = False
    system_summary["exobio_indicators"]["has_codex_biology"] = has_bio

    game_info = {
        "version": last_loadgame.get("gameversion") if last_loadgame else None,
        "build": last_loadgame.get("build") if last_loadgame else None,
        "odyssey": None,  # best-effort; can be inferred later if needed
    }

    commander_hash = hash_commander_name(last_loadgame.get("Commander")) if last_loadgame else None

    return {
        "visit_window": {
            "from_utc": visit_time_min,
            "to_utc": visit_time_max,
            "anchor_timestamp_utc": anchor_ts,
            "anchor_event": anchor.anchor_event.get("event"),
            "journal_anchor_file": anchor.journal_file,
        },
        "game": game_info,
        "commander": {
            "name": None,          # intentionally not stored
            "id_hash": commander_hash,
        },
        "system_summary": system_summary,
        "bodies": {
            "stars": stars,
            "planets": planets,
        },
        "events": {
            # Keep raw events for reproducibility (can be trimmed later if repo size becomes an issue)
            "FSSDiscoveryScan": collected["FSSDiscoveryScan"] or [],
            "SAASignalsFound": collected["SAASignalsFound"] or [],
            "CodexEntry": collected["CodexEntry"] or [],
            "Other": collected["Other"] or [],
            # Raw Scan can be huge; we keep pruned versions above. Keep count here:
            "Scan_count": len(scans),
        },
    }

def prune_scan_event(scan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep a useful subset of Scan fields (still rich), while avoiding extremely verbose nested blobs.
    You can expand this later as needed.
    """
    keep_keys = {
        "timestamp", "event",
        "BodyName", "BodyID",
        "StarSystem", "SystemAddress",
        "DistanceFromArrivalLS",
        "StarType", "Subclass", "StellarMass", "Radius", "SurfaceTemperature", "AbsoluteMagnitude", "Age_MY",
        "Luminosity", "SemiMajorAxis", "Eccentricity", "OrbitalInclination", "Periapsis", "OrbitalPeriod",
        "RotationPeriod", "AxialTilt",
        "PlanetClass", "TerraformState", "Atmosphere", "AtmosphereType", "AtmosphereComposition",
        "Volcanism", "MassEM", "Radius", "SurfaceGravity", "SurfacePressure", "SurfaceTemperature",
        "Landable", "Materials", "Composition", "Rings",
        "Parents",
    }
    out: Dict[str, Any] = {}
    for k, v in scan.items():
        if k in keep_keys:
            out[k] = v
    # Ensure stable keys even if missing
    if "Landable" not in out and "PlanetClass" in out:
        out["Landable"] = None
    return out


# -------------------------
# Main ingest logic
# -------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest one FLAC from ./incoming/, rename/move it, and generate metadata JSON from Elite logs."
    )
    parser.add_argument("system_name", help="Exact in-game system name, e.g. \"Praea Eurl RY-H d10-0\"")
    parser.add_argument("--incoming", default="incoming", help="Incoming folder containing exactly one FLAC (default: incoming)")
    parser.add_argument("--audio-out", default="audio", help="Output folder for renamed FLAC (default: audio)")
    parser.add_argument("--meta-out", default="metadata", help="Output folder for JSON metadata (default: metadata)")
    parser.add_argument("--journals", default=None, help="Elite Dangerous journals directory. Defaults to standard Windows path.")
    parser.add_argument("--max-seconds", type=float, default=60.0, help="Max allowed duration for FLAC (default: 60)")
    args = parser.parse_args()

    repo_root = Path.cwd()
    incoming_dir = (repo_root / args.incoming).resolve()
    audio_dir = (repo_root / args.audio_out).resolve()
    meta_dir = (repo_root / args.meta_out).resolve()

    incoming_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # Find exactly one FLAC in incoming
    flacs = sorted([p for p in incoming_dir.glob("*.flac") if p.is_file()])
    if len(flacs) == 0:
        print(f"[ERROR] No .flac found in {incoming_dir}")
        return 2
    if len(flacs) > 1:
        print(f"[ERROR] More than one .flac found in {incoming_dir}. Please keep exactly one.")
        for p in flacs:
            print(f"  - {p.name}")
        return 2

    src_audio = flacs[0]

    # Duration check
    dur = get_flac_duration_seconds(src_audio)
    if dur is None:
        print("[ERROR] Could not read FLAC duration. Please install dependency:")
        print("        pip install mutagen")
        return 3
    if dur > float(args.max_seconds) + 1e-6:
        print(f"[ERROR] FLAC duration is {dur:.2f}s, which exceeds the limit ({args.max_seconds:.0f}s).")
        return 4

    # Hash for collision-free naming
    full_hash = sha256_file(src_audio)
    short_hash = full_hash[:8]

    system_name_raw = args.system_name.strip()
    system_name_safe = sanitize_system_name(system_name_raw)
    if not system_name_safe:
        print("[ERROR] System name sanitization resulted in an empty name. Please provide a valid system name.")
        return 5

    base_name = f"{system_name_safe}__{short_hash}"
    dst_audio = audio_dir / f"{base_name}.flac"
    dst_meta = meta_dir / f"{base_name}.json"

    # Avoid duplicates (same audio hash)
    if dst_audio.exists() or dst_meta.exists():
        print("[ERROR] A recording with the same hash already exists in the repo:")
        if dst_audio.exists():
            print(f"  - {dst_audio}")
        if dst_meta.exists():
            print(f"  - {dst_meta}")
        print("This usually means the same audio was already ingested.")
        return 6

    # Parse journals
    journal_dir = Path(args.journals).expanduser().resolve() if args.journals else default_elite_journal_dir().resolve()
    journal_files = iter_journal_files(journal_dir)
    if not journal_files:
        print(f"[WARN] No journal files found in: {journal_dir}")
        print("       Metadata will be generated with null fields where appropriate.")
        anchor = None
        extracted = None
    else:
        anchor = find_latest_visit_anchor(journal_files, system_name_raw)
        if anchor is None:
            print(f"[WARN] Could not find a recent visit to system '{system_name_raw}' in journals.")
            extracted = None
        else:
            extracted = extract_events_for_visit(journal_files, anchor)

    # Move audio into place
    shutil.move(str(src_audio), str(dst_audio))

    # Build metadata JSON
    meta: Dict[str, Any] = {
        "recording": {
            "audio_file": dst_audio.name,
            "audio_sha256": full_hash,
            "audio_sha256_short": short_hash,
            "original_filename": src_audio.name,
            "created_utc": utc_now_iso(),
            "duration_s": round(dur, 3),
        },
        "context": {
            "system": {
                "name": system_name_raw,
                "name_sanitized": system_name_safe,
                "address": None,
                "star_pos": None,
            },
            "game": {"version": None, "build": None, "odyssey": None},
            "commander": {"name": None, "id_hash": None},
        },
        "acquisition": {
            "notes": None,
        },
        "visit": {
            "from_utc": None,
            "to_utc": None,
            "anchor_timestamp_utc": None,
            "anchor_event": None,
            "journal_anchor_file": None,
        },
        "system_summary": {
            "body_count": None,
            "non_body_signals": None,
            "stars_count": None,
            "planets_count": None,
            "landables_count": None,
            "exobio_indicators": {"codex_entries_count": None, "has_codex_biology": None},
        },
        "bodies": {
            "stars": [],
            "planets": [],
        },
        "raw": {
            "journal_dir": str(journal_dir),
            "journal_files_used": [p.name for p in journal_files[-10:]] if journal_files else [],
            "events_time_window": {"from_utc": None, "to_utc": None},
        },
    }

    if anchor is not None:
        meta["context"]["system"]["address"] = anchor.system_address
        meta["context"]["system"]["star_pos"] = anchor.star_pos

    if extracted is not None:
        meta["context"]["game"] = extracted.get("game", meta["context"]["game"])
        meta["context"]["commander"] = extracted.get("commander", meta["context"]["commander"])
        vw = extracted.get("visit_window", {})
        meta["visit"]["from_utc"] = vw.get("from_utc")
        meta["visit"]["to_utc"] = vw.get("to_utc")
        meta["visit"]["anchor_timestamp_utc"] = vw.get("anchor_timestamp_utc")
        meta["visit"]["anchor_event"] = vw.get("anchor_event")
        meta["visit"]["journal_anchor_file"] = vw.get("journal_anchor_file")
        meta["system_summary"] = extracted.get("system_summary", meta["system_summary"])
        bodies = extracted.get("bodies", {})
        meta["bodies"]["stars"] = bodies.get("stars", []) or []
        meta["bodies"]["planets"] = bodies.get("planets", []) or []
        meta["raw"]["events_time_window"]["from_utc"] = vw.get("from_utc")
        meta["raw"]["events_time_window"]["to_utc"] = vw.get("to_utc")

    # Write JSON
    with dst_meta.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("✅ Ingest complete")
    print(f"System:   {system_name_raw}")
    print(f"Audio:    {dst_audio.relative_to(repo_root)}")
    print(f"Metadata: {dst_meta.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
