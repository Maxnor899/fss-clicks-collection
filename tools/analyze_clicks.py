#!/usr/bin/env python3
"""
Analyze FSS click recordings already ingested with tools/ingest.py.

Reads:
- ./metadata/*.json (as produced by ingest.py)
- ./audio/*.flac    (resolved via metadata.recording.audio_file)

Writes (default):
- ./analysis/summary.json
- ./analysis/per_file/<SystemNameSafe>__<hash>.json
- ./analysis/labels/<SystemNameSafe>__<hash>__ticks.txt
- ./analysis/labels/<SystemNameSafe>__<hash>__motifs.txt

No graphs. Intended for human audit using timestamps in external tools.

Dependencies:
  pip install numpy soundfile
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


# -------------------------
# Tick detection (same spirit as your test.py)
# -------------------------

@dataclass
class Tick:
    sample_idx: int
    t_s: float
    strength: float  # envelope value at peak (audit/thresholding/debug)

def detect_ticks_with_strength(
    audio: np.ndarray,
    sr: int,
    smooth_ms: float = 5.0,
    min_dist_ms: float = 20.0,
    mad_k: float = 12.0,
) -> Tuple[List[Tick], Dict[str, float]]:
    # accentuer transitoires
    hp = np.concatenate([[0.0], np.diff(audio)])

    # enveloppe + lissage court
    env = np.abs(hp)
    win = max(1, int(sr * (smooth_ms / 1000.0)))
    smooth = np.convolve(env, np.ones(win, dtype=np.float64) / float(win), mode="same")

    # seuil robuste (MAD)
    med = float(np.median(smooth))
    mad = float(np.median(np.abs(smooth - med)) + 1e-12)
    thr = med + float(mad_k) * mad

    # pics avec distance min + montée au max local
    min_dist = int(sr * (min_dist_ms / 1000.0))
    peaks: List[int] = []
    last = -min_dist
    i = 0
    n = len(smooth)

    while i < n:
        if smooth[i] > thr and (i - last) >= min_dist:
            j = i
            while j + 1 < n and smooth[j + 1] >= smooth[j]:
                j += 1
            peaks.append(j)
            last = j
            i = j + 1
        else:
            i += 1

    ticks: List[Tick] = []
    for p in peaks:
        ticks.append(Tick(sample_idx=int(p), t_s=float(p) / float(sr), strength=float(smooth[p])))

    stats = {
        "median": med,
        "mad": mad,
        "threshold": float(thr),
        "smooth_ms": float(smooth_ms),
        "min_dist_ms": float(min_dist_ms),
        "mad_k": float(mad_k),
        "sr": float(sr),
    }
    return ticks, stats


# -------------------------
# Motif matching
# -------------------------

@dataclass
class MotifHit:
    start_s: float
    end_s: float
    ticks_s: List[float]   # len = 7 (if template has 6 dt)
    dt_s: List[float]      # len = 6

def find_motifs(tick_times: List[float], dt_template: np.ndarray, tol: float) -> List[MotifHit]:
    t = np.asarray(tick_times, dtype=np.float64)
    L = int(len(dt_template) + 1)  # 7 ticks if template has 6 dt
    hits: List[MotifHit] = []

    if len(t) < L:
        return hits

    for i in range(0, len(t) - L + 1):
        window = t[i : i + L]
        dt = np.diff(window)
        if np.all(np.abs(dt - dt_template) <= tol):
            hits.append(
                MotifHit(
                    start_s=float(window[0]),
                    end_s=float(window[-1]),
                    ticks_s=[float(x) for x in window.tolist()],
                    dt_s=[float(x) for x in dt.tolist()],
                )
            )
    return hits


# -------------------------
# Energy / RMS helpers
# -------------------------

def mean_square(x: np.ndarray) -> float:
    # energy proxy, stable for ratios
    return float(np.mean(np.square(x, dtype=np.float64), dtype=np.float64) + 1e-18)

def slice_by_time(x: np.ndarray, sr: int, t0: float, t1: float) -> Optional[np.ndarray]:
    n = len(x)
    a = int(round(t0 * sr))
    b = int(round(t1 * sr))
    a = max(0, min(n, a))
    b = max(0, min(n, b))
    if b <= a:
        return None
    return x[a:b]

@dataclass
class WindowDef:
    kind: str  # "tick" or "bg"
    t0_s: float
    t1_s: float

@dataclass
class TickIntensity:
    t_s: float
    sample_idx: int
    energy_tick: Optional[float]
    energy_bg: Optional[float]
    snr_db: Optional[float]
    windows: List[WindowDef]  # which windows were used


def compute_tick_intensity_for_motif(
    x: np.ndarray,
    sr: int,
    motif_ticks: List[float],
    tick_index: int,
    tick_win_ms: float = 20.0,
    tick_pre_ms: float = 5.0,   # 20ms window: [-5ms, +15ms]
    bg_win_ms: float = 10.0,
    bg_margin_ms: float = 1.0,  # stay away from tick centers inside gaps
) -> TickIntensity:
    """
    Tick window: 20ms around tick center (asymmetric -pre/+post).
    Background windows: 10ms centered at the midpoint of the gaps:
      left gap midpoint between tick(k-1) and tick(k)
      right gap midpoint between tick(k) and tick(k+1)
    Valid only if the gap can contain the window with a margin from both tick centers.
    """
    tk = float(motif_ticks[tick_index])
    tick_pre = tick_pre_ms / 1000.0
    tick_len = tick_win_ms / 1000.0
    tick_post = tick_len - tick_pre

    windows_used: List[WindowDef] = []

    # Tick window
    t0_tick = tk - tick_pre
    t1_tick = tk + tick_post
    seg_tick = slice_by_time(x, sr, t0_tick, t1_tick)
    windows_used.append(WindowDef(kind="tick", t0_s=float(t0_tick), t1_s=float(t1_tick)))

    e_tick = mean_square(seg_tick) if seg_tick is not None and len(seg_tick) >= 8 else None

    # Background candidates
    bg_len = bg_win_ms / 1000.0
    margin = bg_margin_ms / 1000.0
    bg_energies: List[Tuple[float, WindowDef]] = []

    # Left gap
    if tick_index > 0:
        t_prev = float(motif_ticks[tick_index - 1])
        gap = tk - t_prev
        # Need enough room: margin from both tick centers + bg_len
        if gap >= (2 * margin + bg_len):
            mid = (t_prev + tk) / 2.0
            t0 = mid - bg_len / 2.0
            t1 = mid + bg_len / 2.0
            # Ensure window stays inside (t_prev+margin, tk-margin)
            if t0 >= (t_prev + margin) and t1 <= (tk - margin):
                seg_bg = slice_by_time(x, sr, t0, t1)
                if seg_bg is not None and len(seg_bg) >= 8:
                    bg_energies.append((mean_square(seg_bg), WindowDef(kind="bg_left", t0_s=float(t0), t1_s=float(t1))))

    # Right gap
    if tick_index < len(motif_ticks) - 1:
        t_next = float(motif_ticks[tick_index + 1])
        gap = t_next - tk
        if gap >= (2 * margin + bg_len):
            mid = (tk + t_next) / 2.0
            t0 = mid - bg_len / 2.0
            t1 = mid + bg_len / 2.0
            if t0 >= (tk + margin) and t1 <= (t_next - margin):
                seg_bg = slice_by_time(x, sr, t0, t1)
                if seg_bg is not None and len(seg_bg) >= 8:
                    bg_energies.append((mean_square(seg_bg), WindowDef(kind="bg_right", t0_s=float(t0), t1_s=float(t1))))

    # Choose background energy
    e_bg: Optional[float] = None
    if bg_energies:
        # Use median of available sides (1 or 2)
        vals = np.array([v for v, _w in bg_energies], dtype=np.float64)
        e_bg = float(np.median(vals))
        for _v, w in bg_energies:
            windows_used.append(w)

    snr_db: Optional[float] = None
    if e_tick is not None and e_bg is not None and e_tick > 0 and e_bg > 0:
        snr_db = 10.0 * math.log10((e_tick + 1e-18) / (e_bg + 1e-18))

    # sample index for audit
    sample_idx = int(round(tk * sr))
    sample_idx = max(0, min(len(x) - 1, sample_idx))

    return TickIntensity(
        t_s=tk,
        sample_idx=sample_idx,
        energy_tick=e_tick,
        energy_bg=e_bg,
        snr_db=snr_db,
        windows=windows_used,
    )


# -------------------------
# IO helpers
# -------------------------

def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def write_labels(path: Path, rows: List[Tuple[float, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for t, label in rows:
            f.write(f"{t:.6f}\t{label}\n")


# -------------------------
# Main analysis per file
# -------------------------

def analyze_one(meta_path: Path, audio_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    meta = read_json(meta_path)

    rec = meta.get("recording", {})
    ctx = meta.get("context", {})
    sysinfo = (ctx.get("system") or {}) if isinstance(ctx, dict) else {}

    audio_file = rec.get("audio_file")
    if not isinstance(audio_file, str) or not audio_file.lower().endswith(".flac"):
        raise RuntimeError(f"Invalid metadata.recording.audio_file in {meta_path.name}")

    flac_path = audio_dir / audio_file
    if not flac_path.exists():
        raise RuntimeError(f"Missing audio file: {flac_path}")

    x, sr = sf.read(str(flac_path))
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float64)

    # Ticks
    ticks, tick_stats = detect_ticks_with_strength(
        x, int(sr),
        smooth_ms=args.smooth_ms,
        min_dist_ms=args.min_dist_ms,
        mad_k=args.mad_k
    )
    tick_times = [t.t_s for t in ticks]

    # Motifs
    dt_template = np.array(args.dt_template, dtype=np.float64)
    motifs = find_motifs(tick_times, dt_template=dt_template, tol=args.tol)

    # Intensity on motif ticks
    motif_intensities: List[Dict[str, Any]] = []
    snr_all: List[float] = []

    for mi, mh in enumerate(motifs):
        tick_ints: List[Dict[str, Any]] = []
        for k in range(len(mh.ticks_s)):
            ti = compute_tick_intensity_for_motif(
                x, int(sr),
                motif_ticks=mh.ticks_s,
                tick_index=k,
                tick_win_ms=args.tick_win_ms,
                tick_pre_ms=args.tick_pre_ms,
                bg_win_ms=args.bg_win_ms,
                bg_margin_ms=args.bg_margin_ms,
            )
            d = {
                "tick_index": k,
                "t_s": ti.t_s,
                "sample_idx": ti.sample_idx,
                "energy_tick": ti.energy_tick,
                "energy_bg": ti.energy_bg,
                "snr_db": ti.snr_db,
                "windows": [{"kind": w.kind, "t0_s": w.t0_s, "t1_s": w.t1_s} for w in ti.windows],
            }
            if ti.snr_db is not None and math.isfinite(ti.snr_db):
                snr_all.append(float(ti.snr_db))
            tick_ints.append(d)

        motif_intensities.append({
            "motif_index": mi,
            "start_s": mh.start_s,
            "end_s": mh.end_s,
            "ticks_s": mh.ticks_s,
            "dt_s": mh.dt_s,
            "ticks": tick_ints,
        })

    Ii_median = float(np.median(snr_all)) if snr_all else None
    Ii_p10 = float(np.percentile(np.array(snr_all), 10)) if snr_all else None
    Ii_p90 = float(np.percentile(np.array(snr_all), 90)) if snr_all else None

    # Per-file report
    report: Dict[str, Any] = {
        "meta_file": meta_path.name,
        "recording": {
            "audio_file": audio_file,
            "audio_sha256": rec.get("audio_sha256"),
            "audio_sha256_short": rec.get("audio_sha256_short"),
            "duration_s": rec.get("duration_s"),
            "sr": int(sr),
        },
        "system": {
            "name": sysinfo.get("name"),
            "name_sanitized": sysinfo.get("name_sanitized"),
            "address": sysinfo.get("address"),
            "star_pos": sysinfo.get("star_pos"),
        },
        "tick_detection": {
            "tick_stats": tick_stats,
            "ticks_count": len(ticks),
            "ticks": [{"t_s": t.t_s, "sample_idx": t.sample_idx, "strength": t.strength} for t in ticks],
        },
        "motif": {
            "dt_template_s": [float(v) for v in dt_template.tolist()],
            "tol_s": float(args.tol),
            "motifs_count": len(motifs),
            "motifs": motif_intensities,
        },
        "intensity_summary": {
            "snr_db_count": len(snr_all),
            "Ii_db_median": Ii_median,
            "Ii_db_p10": Ii_p10,
            "Ii_db_p90": Ii_p90,
            "params": {
                "tick_win_ms": float(args.tick_win_ms),
                "tick_pre_ms": float(args.tick_pre_ms),
                "bg_win_ms": float(args.bg_win_ms),
                "bg_margin_ms": float(args.bg_margin_ms),
            },
        },
    }

    return report


# -------------------------
# CLI / Orchestration
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze ingested FSS click recordings (ticks, motifs, intensity).")

    p.add_argument("--metadata", default="metadata", help="Metadata folder (default: metadata)")
    p.add_argument("--audio", default="audio", help="Audio folder (default: audio)")
    p.add_argument("--out", default="analysis", help="Output folder (default: analysis)")

    # Tick detection params
    p.add_argument("--smooth-ms", type=float, default=5.0)
    p.add_argument("--min-dist-ms", type=float, default=20.0)
    p.add_argument("--mad-k", type=float, default=12.0)

    # Motif template + tolerance
    # default = template from your tictac4
    p.add_argument(
        "--dt-template",
        type=float,
        nargs="+",
        default=[0.642, 0.643, 0.514, 0.121, 0.646, 0.634],
        help="Motif dt template (seconds), 6 values for 7 ticks",
    )
    p.add_argument("--tol", type=float, default=0.05, help="Tolerance on each dt (seconds)")

    # Intensity window params
    p.add_argument("--tick-win-ms", type=float, default=20.0)
    p.add_argument("--tick-pre-ms", type=float, default=5.0)      # -5ms / +15ms
    p.add_argument("--bg-win-ms", type=float, default=10.0)
    p.add_argument("--bg-margin-ms", type=float, default=1.0)

    return p.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()

    meta_dir = (repo_root / args.metadata).resolve()
    audio_dir = (repo_root / args.audio).resolve()
    out_dir = (repo_root / args.out).resolve()

    per_file_dir = out_dir / "per_file"
    labels_dir = out_dir / "labels"

    meta_files = sorted([p for p in meta_dir.glob("*.json") if p.is_file()])
    if not meta_files:
        print(f"[ERROR] No metadata json files found in: {meta_dir}")
        return 2

    summary_rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for mp in meta_files:
        try:
            report = analyze_one(mp, audio_dir=audio_dir, args=args)

            sys_name_safe = report["system"]["name_sanitized"] or mp.stem
            short_hash = report["recording"]["audio_sha256_short"] or "nohash"
            base = f"{sys_name_safe}__{short_hash}"

            # write per-file json
            write_json(per_file_dir / f"{base}.json", report)

            # write labels: all ticks
            tick_labels = [(float(t["t_s"]), "tick") for t in report["tick_detection"]["ticks"]]
            write_labels(labels_dir / f"{base}__ticks.txt", tick_labels)

            # write labels: motif ticks only (with indices)
            motif_rows: List[Tuple[float, str]] = []
            for m in report["motif"]["motifs"]:
                mi = m["motif_index"]
                for t in m["ticks"]:
                    k = t["tick_index"]
                    motif_rows.append((float(t["t_s"]), f"motif{mi}_tick{k}"))
            write_labels(labels_dir / f"{base}__motifs.txt", motif_rows)

            # add to summary
            summary_rows.append({
                "base": base,
                "meta_file": report["meta_file"],
                "audio_file": report["recording"]["audio_file"],
                "system_name": report["system"]["name"],
                "system_address": report["system"]["address"],
                "star_pos": report["system"]["star_pos"],
                "ticks_count": report["tick_detection"]["ticks_count"],
                "motifs_count": report["motif"]["motifs_count"],
                "snr_db_count": report["intensity_summary"]["snr_db_count"],
                "Ii_db_median": report["intensity_summary"]["Ii_db_median"],
                "Ii_db_p10": report["intensity_summary"]["Ii_db_p10"],
                "Ii_db_p90": report["intensity_summary"]["Ii_db_p90"],
            })

            print(f"✅ {base}: ticks={report['tick_detection']['ticks_count']} motifs={report['motif']['motifs_count']} Ii={report['intensity_summary']['Ii_db_median']} dB")

        except Exception as e:
            errors.append({"meta_file": mp.name, "error": str(e)})
            print(f"[WARN] {mp.name}: {e}")

    # write summary
    summary = {
        "run": {
            "metadata_dir": str(meta_dir),
            "audio_dir": str(audio_dir),
            "out_dir": str(out_dir),
            "params": {
                "smooth_ms": args.smooth_ms,
                "min_dist_ms": args.min_dist_ms,
                "mad_k": args.mad_k,
                "dt_template": args.dt_template,
                "tol": args.tol,
                "tick_win_ms": args.tick_win_ms,
                "tick_pre_ms": args.tick_pre_ms,
                "bg_win_ms": args.bg_win_ms,
                "bg_margin_ms": args.bg_margin_ms,
            }
        },
        "files_ok": len(summary_rows),
        "files_error": len(errors),
        "errors": errors,
        "systems": summary_rows,
    }
    write_json(out_dir / "summary.json", summary)

    print(f"\nDone. Wrote: {out_dir / 'summary.json'}")
    print(f"Per-file reports: {per_file_dir}")
    print(f"Labels: {labels_dir}")

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
