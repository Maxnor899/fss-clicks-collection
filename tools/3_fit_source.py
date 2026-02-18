#!/usr/bin/env python3
"""
tools/fit_source.py
===================

Fit a hypothetical "FSS click source" from per-system intensity measurements.

This script is designed to plug into the repo layout used by `tools/ingest.py`
and `tools/analyze_clicks.py`.

It reads:
  - ./analysis/summary.json  (produced by tools/analyze_clicks.py)
  - ./metadata/*.json       (optional: for Model C covariates)

It writes:
  - ./analysis/fit_source__model_*.json
  - ./analysis/fit_source__model_*__bootstrap.json (unless --no-bootstrap)

NEW (optional):
  - --use-ii-focus uses Ii_focus_db from analysis_Ii_ref/Ii_focus_summary.csv
    instead of Ii_db_median from analysis/summary.json.

Notes:
- The fit is diagnostic: it will find a numerical optimum, but the stability
  (bootstrap) and residuals determine whether the model is meaningful.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _norm(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))


def _extract_short_hash(base: Optional[str], audio_file: Optional[str]) -> Optional[str]:
    """
    Try to extract a stable short hash from base/audio_file.
    Expected patterns:
      base: "<SystemSafe>__<hash>"
      audio_file: "<SystemSafe>__<hash>.flac"
    """
    if isinstance(audio_file, str):
        m = re.search(r"__([0-9a-fA-F]{6,})\.", audio_file)
        if m:
            return m.group(1).lower()
    if isinstance(base, str):
        m = re.search(r"__([0-9a-fA-F]{6,})$", base)
        if m:
            return m.group(1).lower()
    return None


# ---------------------------------------------------------------------------
# Data structure for one system
# ---------------------------------------------------------------------------

class SystemPoint:
    """
    Represents one system measurement used for fitting.

    Fields come primarily from analysis/summary.json.

    Attributes
    ----------
    name:
        Human-readable system name (from summary.json).
    pos:
        np.ndarray shape (3,) coordinates in ly (from summary.json -> star_pos).
    snr_db:
        Ii_db_median (dB). In analyze_clicks.py, this is 10*log10(E_tick/E_bg).
        (May be replaced by Ii_focus_db when --use-ii-focus is enabled.)
    snr_p10 / snr_p90:
        Percentiles in dB for dispersion (optional).
    base:
        A stable identifier produced in analyze_clicks.py summary: "<SystemSafe>__<hash>".
        This is extremely useful as a joining key.
    audio_file:
        The canonical audio filename in ./audio, usually "<SystemSafe>__<hash>.flac".
        We use it to recover the short hash when joining metadata.
    short_hash:
        Extracted from base/audio_file when possible. Used for robust metadata lookup.

    Covariates (Model C)
    --------------------
    primary_star_type:
        ED StarType string for the primary star (if present).
    primary_age_my:
        Age_MY of the primary star (if present).
    stars_count:
        Number of star bodies found in metadata.
    has_tts:
        Whether any star type begins with "TTS" (T Tauri).
    is_active:
        Coarse categorisation from StarType: calm vs active.
    """

    name: str
    pos: np.ndarray
    snr_db: float
    snr_p10: Optional[float]
    snr_p90: Optional[float]
    base: Optional[str] = None
    audio_file: Optional[str] = None
    short_hash: Optional[str] = None

    # Covariates (Model C)
    primary_star_type: Optional[str] = None
    primary_age_my: Optional[float] = None
    stars_count: Optional[int] = None
    has_tts: Optional[bool] = None
    is_active: Optional[str] = None  # "calm" / "active" / None

    def __init__(
        self,
        name: str,
        pos: np.ndarray,
        snr_db: float,
        snr_p10: Optional[float],
        snr_p90: Optional[float],
        base: Optional[str] = None,
        audio_file: Optional[str] = None,
        short_hash: Optional[str] = None,
    ) -> None:
        self.name = name
        self.pos = pos
        self.snr_db = snr_db
        self.snr_p10 = snr_p10
        self.snr_p90 = snr_p90
        self.base = base
        self.audio_file = audio_file
        self.short_hash = short_hash


# ---------------------------------------------------------------------------
# Input: summary.json
# ---------------------------------------------------------------------------

def load_summary(summary_path: Path) -> List[SystemPoint]:
    """
    Load analysis/summary.json and extract usable points.

    Summary rows come from analyze_clicks.py and typically include:
      base, audio_file, system_name, star_pos, Ii_db_median, Ii_db_p10, Ii_db_p90, ...

    We keep only entries with valid star_pos and Ii_db_median.
    """
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    points: List[SystemPoint] = []
    for s in summary.get("systems", []):
        pos = s.get("star_pos")
        snr = s.get("Ii_db_median")
        if not (isinstance(pos, list) and len(pos) == 3 and isinstance(snr, (int, float))):
            continue

        base = s.get("base")
        audio_file = s.get("audio_file")
        sh = _extract_short_hash(base, audio_file)

        points.append(SystemPoint(
            name=str(s.get("system_name") or base or "?"),
            pos=np.array(pos, dtype=np.float64),
            snr_db=float(snr),
            snr_p10=float(s["Ii_db_p10"]) if isinstance(s.get("Ii_db_p10"), (int, float)) else None,
            snr_p90=float(s["Ii_db_p90"]) if isinstance(s.get("Ii_db_p90"), (int, float)) else None,
            base=str(base) if isinstance(base, str) else None,
            audio_file=str(audio_file) if isinstance(audio_file, str) else None,
            short_hash=sh,
        ))
    return points


# ---------------------------------------------------------------------------
# Optional: override Ii using Ii_focus (slots 5+6) computed from click_events
# ---------------------------------------------------------------------------

def load_ii_focus_csv(path: Path) -> Dict[str, float]:
    """Load Ii_focus_summary.csv into a lookup map.

    Expected columns:
      - file (e.g. 'Blo_Thae_...__hash.flac')
      - Ii_focus_db (float)

    We store multiple keys for robust matching:
      - exact filename
      - filename stem (without extension)
    """
    m: Dict[str, float] = {}
    if not path.exists():
        return m

    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            fn = (row.get("file") or "").strip()
            val = (row.get("Ii_focus_db") or "").strip()
            if not fn or not val:
                continue
            try:
                v = float(val)
            except Exception:
                continue
            m[fn] = v
            stem = Path(fn).stem
            if stem:
                m[stem] = v
    return m


def apply_ii_focus(points: List[SystemPoint], ii_focus: Dict[str, float]) -> Dict[str, int]:
    """Replace SystemPoint.snr_db with Ii_focus_db when available.

    Matching priority per point:
      1) audio_file exact (e.g. '<base>.flac')
      2) audio_file stem
      3) base exact (e.g. '<SystemSafe>__<hash>')
      4) base + '.flac'
      5) short_hash (last resort)

    Returns counts: replaced, missing.
    """
    replaced = 0
    missing = 0

    for p in points:
        keys: List[str] = []
        if isinstance(p.audio_file, str) and p.audio_file:
            keys.append(p.audio_file)
            keys.append(Path(p.audio_file).stem)
        if isinstance(p.base, str) and p.base:
            keys.append(p.base)
            keys.append(p.base + ".flac")
        if isinstance(p.short_hash, str) and p.short_hash:
            keys.append(p.short_hash)

        v = None
        for k in keys:
            if k in ii_focus:
                v = ii_focus[k]
                break

        if v is None:
            missing += 1
            continue

        p.snr_db = float(v)
        # Percentiles are not defined for Ii_focus unless you computed them separately.
        p.snr_p10 = None
        p.snr_p90 = None
        replaced += 1

    return {"replaced": replaced, "missing": missing}


# ---------------------------------------------------------------------------
# Metadata covariates (Model C)
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _guess_primary_star(body_list: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Heuristic: primary star is first body with StarType or first body type 'Star'.
    Metadata formats vary a bit; we keep it robust.
    """
    for b in body_list:
        if isinstance(b, dict) and ("StarType" in b or b.get("type") == "Star"):
            return b
    return None


def load_covariates(points: List[SystemPoint], meta_dir: Path) -> None:
    """
    Best-effort join of metadata covariates onto SystemPoint.

    We try by:
      - short_hash -> metadata/<short_hash>.json
      - base hash patterns
    """
    if not meta_dir.exists():
        return

    # Build map from possible keys -> SystemPoint
    by_hash: Dict[str, SystemPoint] = {}
    for p in points:
        if p.short_hash:
            by_hash[p.short_hash.lower()] = p

    # Iterate json files
    for jpath in meta_dir.glob("*.json"):
        key = jpath.stem.lower()
        p = by_hash.get(key)
        if p is None:
            continue

        meta = _load_json(jpath)
        if not isinstance(meta, dict):
            continue

        bodies = meta.get("bodies")
        if not isinstance(bodies, list):
            continue

        p.stars_count = sum(1 for b in bodies if isinstance(b, dict) and ("StarType" in b or b.get("type") == "Star"))
        p.has_tts = any(isinstance(b, dict) and isinstance(b.get("StarType"), str) and b["StarType"].startswith("TTS") for b in bodies)

        primary = _guess_primary_star(bodies)
        if isinstance(primary, dict):
            st = primary.get("StarType")
            if isinstance(st, str):
                p.primary_star_type = st
                # Very coarse "activity" proxy: keep stable & debatable.
                p.is_active = "active" if ("TTS" in st or "AeBe" in st) else "calm"

            age = primary.get("Age_MY")
            if isinstance(age, (int, float)):
                p.primary_age_my = float(age)


# ---------------------------------------------------------------------------
# Models and fitting
# ---------------------------------------------------------------------------

def predict_I_db(alpha: float, r0: float, src: np.ndarray, pos: np.ndarray, gain_db: float = 0.0) -> float:
    """
    Simple attenuation model in dB:
      I_db = gain_db + C - 10*alpha*log10(r + r0)

    C is absorbed into gain_db (per-system gain for Model B) or into a global bias.
    """
    r = _norm(pos - src)
    return float(gain_db - 10.0 * alpha * math.log10(r + r0))


def fit_model_A(points: List[SystemPoint], alpha_bounds: Tuple[float, float], source_reg: float) -> Dict[str, Any]:
    """
    Model A:
      Ii_db ≈ b - 10*alpha*log10(r + r0)

    Fit parameters:
      src (x,y,z), alpha, b, r0

    We use a simple multi-start random search + local refinement by coordinate perturbation.
    (Kept intentionally dependency-free.)
    """
    y = np.array([p.snr_db for p in points], dtype=np.float64)
    X = np.stack([p.pos for p in points], axis=0)

    # init bounds
    mins = X.min(axis=0) - 200.0
    maxs = X.max(axis=0) + 200.0

    alpha_min, alpha_max = alpha_bounds
    best = None

    def loss(params: Dict[str, float], src: np.ndarray) -> float:
        alpha = params["alpha"]
        b = params["b"]
        r0 = params["r0"]
        pred = np.array([b - 10.0 * alpha * math.log10(_norm(X[i] - src) + r0) for i in range(len(points))], dtype=np.float64)
        # L2 + mild source regularization
        res = pred - y
        return float(np.mean(res * res) + source_reg * float(np.sum(src * src)))

    # Random multi-start
    for _ in range(250):
        src = np.array([
            random.uniform(mins[0], maxs[0]),
            random.uniform(mins[1], maxs[1]),
            random.uniform(mins[2], maxs[2]),
        ], dtype=np.float64)

        alpha = random.uniform(alpha_min, alpha_max)
        r0 = 1.0
        # solve b analytically for fixed src, alpha, r0
        base = np.array([-10.0 * alpha * math.log10(_norm(X[i] - src) + r0) for i in range(len(points))], dtype=np.float64)
        b = float(np.mean(y - base))

        params = {"alpha": alpha, "b": b, "r0": r0}
        L = loss(params, src)

        if best is None or L < best["loss"]:
            best = {"loss": L, "src": src.copy(), **params}

    assert best is not None

    # Local refinement: small coordinate perturbations
    src = best["src"].copy()
    alpha = best["alpha"]
    r0 = best["r0"]

    step = 80.0
    for _ in range(120):
        improved = False
        for axis in range(3):
            for sign in (-1.0, 1.0):
                trial_src = src.copy()
                trial_src[axis] += sign * step
                base = np.array([-10.0 * alpha * math.log10(_norm(X[i] - trial_src) + r0) for i in range(len(points))], dtype=np.float64)
                b = float(np.mean(y - base))
                params = {"alpha": alpha, "b": b, "r0": r0}
                L = loss(params, trial_src)
                if L < best["loss"]:
                    best.update({"loss": L, "src": trial_src.copy(), "b": b})
                    src = trial_src
                    improved = True
        if not improved:
            step *= 0.6
            if step < 1.0:
                break

    # Final residuals
    base = np.array([-10.0 * best["alpha"] * math.log10(_norm(X[i] - best["src"]) + best["r0"]) for i in range(len(points))], dtype=np.float64)
    pred = best["b"] + base
    res = pred - y

    return {
        "model": "A",
        "alpha": float(best["alpha"]),
        "b": float(best["b"]),
        "r0": float(best["r0"]),
        "src": [float(x) for x in best["src"]],
        "loss": float(best["loss"]),
        "rmse": float(math.sqrt(float(np.mean(res * res)))),
        "residuals_db": {points[i].name: float(res[i]) for i in range(len(points))},
    }


def fit_model_B(points: List[SystemPoint], alpha_bounds: Tuple[float, float], lambda_b: float, source_reg: float) -> Dict[str, Any]:
    """
    Model B:
      Ii_db ≈ g_i + b - 10*alpha*log10(r + r0)
    with per-system gain g_i regularized (L2) by lambda_b.

    This model is more flexible and can overfit. Use bootstrap to assess stability.
    """
    y = np.array([p.snr_db for p in points], dtype=np.float64)
    X = np.stack([p.pos for p in points], axis=0)

    mins = X.min(axis=0) - 200.0
    maxs = X.max(axis=0) + 200.0

    alpha_min, alpha_max = alpha_bounds
    n = len(points)
    best = None

    def solve_g_b(base: np.ndarray) -> Tuple[np.ndarray, float]:
        # minimize ||(g + b + base) - y||^2 + lambda_b*||g||^2
        # For fixed b: g_i = (y_i - b - base_i) / (1 + lambda_b)
        # Then choose b to minimize residual; solve by derivative (closed form).
        denom = 1.0 + lambda_b
        # b optimal:
        # residual_i = ( (y_i - b - base_i)/denom + b + base_i ) - y_i
        # Simplify -> linear in b. We'll compute b by least squares on expanded form.
        A = (1.0 - 1.0/denom)
        # pred = y_i - (y_i - b - base_i)/denom  (equivalent)
        # Let's just compute b by minimizing MSE numerically with closed form:
        # pred_i = (y_i - b - base_i)/denom + b + base_i
        # pred_i = y_i/denom - b/denom - base_i/denom + b + base_i
        # pred_i = y_i/denom + b*(1 - 1/denom) + base_i*(1 - 1/denom)
        # residual_i = pred_i - y_i = y_i*(1/denom - 1) + A*b + A*base_i
        # minimize sum (A*b + const_i)^2 -> b = -mean(const_i)/A
        const = y * (1.0/denom - 1.0) + A * base
        if abs(A) < 1e-12:
            b = 0.0
        else:
            b = float(-np.mean(const) / A)
        g = (y - b - base) / denom
        return g, b

    def loss(alpha: float, src: np.ndarray, r0: float) -> Tuple[float, np.ndarray, float]:
        base = np.array([-10.0 * alpha * math.log10(_norm(X[i] - src) + r0) for i in range(n)], dtype=np.float64)
        g, b = solve_g_b(base)
        pred = g + b + base
        res = pred - y
        L = float(np.mean(res * res) + lambda_b * float(np.mean(g * g)) + source_reg * float(np.sum(src * src)))
        return L, g, b

    for _ in range(250):
        src = np.array([
            random.uniform(mins[0], maxs[0]),
            random.uniform(mins[1], maxs[1]),
            random.uniform(mins[2], maxs[2]),
        ], dtype=np.float64)
        alpha = random.uniform(alpha_min, alpha_max)
        r0 = 1.0
        L, g, b = loss(alpha, src, r0)
        if best is None or L < best["loss"]:
            best = {"loss": L, "src": src.copy(), "alpha": alpha, "r0": r0, "b": float(b), "g": g.copy()}

    assert best is not None

    # Final residuals
    base = np.array([-10.0 * best["alpha"] * math.log10(_norm(X[i] - best["src"]) + best["r0"]) for i in range(n)], dtype=np.float64)
    pred = best["g"] + best["b"] + base
    res = pred - y

    return {
        "model": "B",
        "alpha": float(best["alpha"]),
        "b": float(best["b"]),
        "r0": float(best["r0"]),
        "src": [float(x) for x in best["src"]],
        "loss": float(best["loss"]),
        "rmse": float(math.sqrt(float(np.mean(res * res)))),
        "per_system_gain_db": {points[i].name: float(best["g"][i]) for i in range(n)},
        "residuals_db": {points[i].name: float(res[i]) for i in range(n)},
    }


def fit_model_C(points: List[SystemPoint], alpha_bounds: Tuple[float, float], source_reg: float) -> Dict[str, Any]:
    """
    Model C (lightweight covariate extension):
      Ii_db ≈ b + w_age*log1p(age) + w_active*I(active) + w_tts*I(TTS) - 10*alpha*log10(r+r0)

    This is intentionally small and dependency-free; it is NOT meant as a full GLM.
    We do a simple ridge-ish solve for linear terms given (src, alpha, r0).
    """
    y = np.array([p.snr_db for p in points], dtype=np.float64)
    Xpos = np.stack([p.pos for p in points], axis=0)

    mins = Xpos.min(axis=0) - 200.0
    maxs = Xpos.max(axis=0) + 200.0

    alpha_min, alpha_max = alpha_bounds
    n = len(points)

    # Build covariate matrix (n x k)
    age = np.array([float(p.primary_age_my) if isinstance(p.primary_age_my, (int, float)) else 0.0 for p in points], dtype=np.float64)
    x_age = np.log1p(age)
    x_active = np.array([1.0 if p.is_active == "active" else 0.0 for p in points], dtype=np.float64)
    x_tts = np.array([1.0 if p.has_tts else 0.0 for p in points], dtype=np.float64)

    # Add intercept
    Z = np.stack([np.ones(n, dtype=np.float64), x_age, x_active, x_tts], axis=1)  # (n,4)

    ridge = 1e-2
    best = None

    def solve_lin(base: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Solve for beta in (Z beta + base) ≈ y  with ridge
        # beta = argmin ||Z beta - (y - base)||^2 + ridge||beta||^2
        rhs = (y - base).reshape(-1, 1)
        A = Z.T @ Z + ridge * np.eye(Z.shape[1], dtype=np.float64)
        b = Z.T @ rhs
        beta = np.linalg.solve(A, b).reshape(-1)
        pred = (Z @ beta) + base
        return beta, pred

    def loss(alpha: float, src: np.ndarray, r0: float) -> Tuple[float, np.ndarray, np.ndarray]:
        base = np.array([-10.0 * alpha * math.log10(_norm(Xpos[i] - src) + r0) for i in range(n)], dtype=np.float64)
        beta, pred = solve_lin(base)
        res = pred - y
        L = float(np.mean(res * res) + source_reg * float(np.sum(src * src)))
        return L, beta, res

    for _ in range(250):
        src = np.array([
            random.uniform(mins[0], maxs[0]),
            random.uniform(mins[1], maxs[1]),
            random.uniform(mins[2], maxs[2]),
        ], dtype=np.float64)
        alpha = random.uniform(alpha_min, alpha_max)
        r0 = 1.0
        L, beta, res = loss(alpha, src, r0)
        if best is None or L < best["loss"]:
            best = {"loss": L, "src": src.copy(), "alpha": alpha, "r0": r0, "beta": beta.copy(), "res": res.copy()}

    assert best is not None

    return {
        "model": "C",
        "alpha": float(best["alpha"]),
        "r0": float(best["r0"]),
        "src": [float(x) for x in best["src"]],
        "loss": float(best["loss"]),
        "rmse": float(math.sqrt(float(np.mean(best["res"] * best["res"])))),
        "beta": {
            "intercept_b": float(best["beta"][0]),
            "w_log1p_age": float(best["beta"][1]),
            "w_active": float(best["beta"][2]),
            "w_tts": float(best["beta"][3]),
        },
        "residuals_db": {points[i].name: float(best["res"][i]) for i in range(n)},
    }


def bootstrap(points: List[SystemPoint], fit_fn, n_iter: int) -> Dict[str, Any]:
    """
    Simple bootstrap: resample systems with replacement and refit, collect src positions and alpha.
    """
    if n_iter <= 0:
        return {"n": 0, "samples": []}

    samples = []
    for _ in range(n_iter):
        sample = [points[random.randrange(len(points))] for _ in range(len(points))]
        try:
            res = fit_fn(sample)
            samples.append({
                "src": res.get("src"),
                "alpha": res.get("alpha"),
                "rmse": res.get("rmse"),
            })
        except Exception:
            continue

    return {"n": len(samples), "samples": samples}


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit a hypothetical source from analysis/summary.json and system coordinates."
    )
    p.add_argument("--summary", default="analysis/summary.json", help="Path to analysis/summary.json")
    p.add_argument("--metadata", default="metadata", help="Metadata folder (for Model C covariates)")
    p.add_argument("--out", default="analysis", help="Output folder (default: analysis)")
    p.add_argument("--use-ii-focus", action="store_true",
                   help="Use Ii_focus_db from analysis_Ii_ref/Ii_focus_summary.csv instead of Ii_db_median from summary.json")
    p.add_argument("--ii-focus-csv", default="analysis_Ii_ref/Ii_focus_summary.csv",
                   help="Path to Ii_focus_summary.csv (default: analysis_Ii_ref/Ii_focus_summary.csv)")
    p.add_argument("--model", choices=["A", "B", "C", "all"], default="all", help="Which model(s) to run")
    p.add_argument("--lambda-b", type=float, default=1.0, help="Regularization weight for Model B gains")
    p.add_argument("--no-bootstrap", action="store_true", help="Skip bootstrap analysis")
    p.add_argument("--bootstrap-n", type=int, default=200, help="Bootstrap iterations (resampling)")
    p.add_argument("--min-systems", type=int, default=4, help="Minimum systems to attempt a fit")
    p.add_argument("--alpha-min", type=float, default=0.5, help="Min alpha bound (soft)")
    p.add_argument("--alpha-max", type=float, default=4.0, help="Max alpha bound (soft)")
    p.add_argument("--source-reg", type=float, default=0.0, help="L2 regularization on source position (soft)")

    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    _seed_everything(args.seed)

    repo_root = Path.cwd()

    summary_path = (repo_root / args.summary).resolve()
    meta_dir = (repo_root / args.metadata).resolve()
    out_dir = (repo_root / args.out).resolve()

    if not summary_path.exists():
        print(f"[ERROR] summary.json not found: {summary_path}")
        print("        Run tools/analyze_clicks.py first.")
        return 2

    points = load_summary(summary_path)

    if args.use_ii_focus:
        ii_focus_path = (repo_root / args.ii_focus_csv).resolve()
        ii_map = load_ii_focus_csv(ii_focus_path)
        if not ii_map:
            print(f"[WARN] --use-ii-focus set, but Ii_focus CSV not found or empty: {ii_focus_path}")
            print("       Falling back to Ii_db_median from summary.json.")
        else:
            rep = apply_ii_focus(points, ii_map)
            print(f"Using Ii_focus_db from: {ii_focus_path}")
            print(f"  replaced={rep['replaced']}  missing={rep['missing']} (fallback: Ii_db_median for missing)")

    if len(points) < args.min_systems:
        print(f"[ERROR] Only {len(points)} usable systems. Need at least {args.min_systems}.")
        return 3

    # Load covariates (best-effort). This is required only for Model C,
    # but harmless for A/B.
    load_covariates(points, meta_dir)
    n_with_star = sum(1 for p in points if p.primary_star_type is not None)

    print(f"Loaded {len(points)} systems from summary.json")
    print(f"Covariates available for {n_with_star}/{len(points)} systems (metadata join best-effort).")

    print("\nSystems overview:")
    for p in points[:min(12, len(points))]:
        extra = []
        if p.primary_star_type is not None:
            extra.append(f"StarType={p.primary_star_type}")
        if p.primary_age_my is not None:
            extra.append(f"Age_MY={p.primary_age_my:.0f}")
        if p.is_active is not None:
            extra.append(f"active={p.is_active}")
        if p.has_tts is not None:
            extra.append(f"TTS={'yes' if p.has_tts else 'no'}")
        ex = ("  " + "  ".join(extra)) if extra else ""
        print(f"  {p.name:40s} Ii={p.snr_db:6.2f} dB  pos=[{p.pos[0]:.0f},{p.pos[1]:.0f},{p.pos[2]:.0f}]{ex}")

    alpha_bounds = (args.alpha_min, args.alpha_max)

    results = []

    if args.model in ("A", "all"):
        resA = fit_model_A(points, alpha_bounds=alpha_bounds, source_reg=args.source_reg)
        results.append(resA)
        write_json(out_dir / "fit_source__model_A.json", resA)

        if not args.no_bootstrap:
            bootA = bootstrap(points, lambda pts: fit_model_A(pts, alpha_bounds=alpha_bounds, source_reg=args.source_reg), args.bootstrap_n)
            write_json(out_dir / "fit_source__model_A__bootstrap.json", bootA)

        print(f"\nModel A: rmse={resA['rmse']:.3f}  alpha={resA['alpha']:.3f}  src={resA['src']}")

    if args.model in ("B", "all"):
        resB = fit_model_B(points, alpha_bounds=alpha_bounds, lambda_b=args.lambda_b, source_reg=args.source_reg)
        results.append(resB)
        write_json(out_dir / "fit_source__model_B.json", resB)

        if not args.no_bootstrap:
            bootB = bootstrap(points, lambda pts: fit_model_B(pts, alpha_bounds=alpha_bounds, lambda_b=args.lambda_b, source_reg=args.source_reg), args.bootstrap_n)
            write_json(out_dir / "fit_source__model_B__bootstrap.json", bootB)

        print(f"\nModel B: rmse={resB['rmse']:.3f}  alpha={resB['alpha']:.3f}  src={resB['src']}  lambda_b={args.lambda_b}")

    if args.model in ("C", "all"):
        resC = fit_model_C(points, alpha_bounds=alpha_bounds, source_reg=args.source_reg)
        results.append(resC)
        write_json(out_dir / "fit_source__model_C.json", resC)

        if not args.no_bootstrap:
            bootC = bootstrap(points, lambda pts: fit_model_C(pts, alpha_bounds=alpha_bounds, source_reg=args.source_reg), args.bootstrap_n)
            write_json(out_dir / "fit_source__model_C__bootstrap.json", bootC)

        print(f"\nModel C: rmse={resC['rmse']:.3f}  alpha={resC['alpha']:.3f}  src={resC['src']}")

    if results:
        write_json(out_dir / "fit_source__results_all.json", {"results": results})

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
