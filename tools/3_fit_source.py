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
  - ./analysis/source_fit.json
  - ./analysis/source_bootstrap.json  (unless --no-bootstrap)

Why this exists
---------------
`tools/analyze_clicks.py` produces, for each system recording:
  - the system coordinates X_i = (x,y,z) (from metadata)
  - an intensity summary Ii_db_median (in dB), derived from energy ratio tick/background

This script takes those (X_i, I_i) pairs and attempts to fit a "source" S
under progressively richer models:

Model A — Isotropic point source (baseline)
  r_i = K / ||X_i - S||^alpha

Model B — Source + per-system gain (regularized)
  r_i = g_i * K / ||X_i - S||^alpha
  g_i is free per system, but penalized to remain near 1 (in log10 space).
  This explicitly acknowledges that the "audio scene" may scale intensities.

Model C — Source + covariates (requires metadata)
  log10(g_i) is explained by stellar/environment covariates, rather than being free.
  This is more interpretable than Model B, but depends on what your metadata contains.

Important scientific note (the "honesty clause")
------------------------------------------------
This script WILL ALWAYS find "some" best-fit source if you have >=4 points,
even if the model is wrong. Therefore, the actual value is in:
  - residuals per system
  - stability tests (bootstrap / leave-one-out)
  - comparing model A vs B vs C
If Model A is unstable or has huge residuals, that's not failure: it's evidence
the isotropic assumption is inadequate.

Dependencies
------------
  pip install numpy scipy

Usage
-----
  python tools/fit_source.py
  python tools/fit_source.py --model A
  python tools/fit_source.py --model B --lambda-b 1.0
  python tools/fit_source.py --model C
  python tools/fit_source.py --no-bootstrap
  python tools/fit_source.py --bootstrap-n 300
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
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

    # Covariates (filled by load_covariates)
    primary_star_type: Optional[str] = None
    primary_age_my: Optional[float] = None
    stars_count: int = 1
    has_tts: bool = False
    is_active: Optional[bool] = None

    @property
    def r(self) -> float:
        """
        Convert dB energy ratio to a linear ratio.

        Because analyze_clicks.py defines Ii_db as:
            Ii_db = 10*log10(E_tick / E_bg)
        the corresponding linear ratio is:
            r = 10**(Ii_db / 10)
        """
        return 10.0 ** (self.snr_db / 10.0)

    @property
    def log10_r(self) -> float:
        """Log10 of the linear ratio, used for stable fitting in log space."""
        return math.log10(max(self.r, 1e-12))


# ---------------------------------------------------------------------------
# Loading summary.json
# ---------------------------------------------------------------------------

def _extract_short_hash(base: Optional[str], audio_file: Optional[str]) -> Optional[str]:
    """
    Try to recover the short hash from either:
      - base = "<SystemSafe>__<hash>"
      - audio_file = "<SystemSafe>__<hash>.flac"

    This is used as a robust join key into metadata, because names can vary slightly.
    """
    if isinstance(base, str) and "__" in base:
        try:
            return base.split("__")[-1]
        except Exception:
            pass
    if isinstance(audio_file, str) and "__" in audio_file:
        try:
            tail = audio_file.split("__")[-1]
            if tail.lower().endswith(".flac"):
                tail = tail[:-5]
            return tail
        except Exception:
            pass
    return None


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
# Metadata covariates (Model C)
# ---------------------------------------------------------------------------

def _classify_star_type(star_type: str) -> str:
    """
    Coarse calm/active categorisation.

    This is NOT a scientific taxonomy. It's a pragmatic "does this look like a
    very young / massive / strongly active / exotic stellar environment?" flag
    to explain potential audio-scene scaling.
    """
    active_prefixes = {"O", "B", "TTS", "AEBE", "W", "WN", "WC", "WO", "WNC", "CS", "C", "CN"}
    t = star_type.strip().upper()
    for a in active_prefixes:
        if t.startswith(a):
            return "active"
    return "calm"


def build_metadata_index(meta_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Build a metadata index ONCE.

    Keys we store (when available):
      - context.system.name
      - context.system.name_sanitized
      - recording.audio_sha256_short  (best key)
      - filename stem (e.g. "<SystemSafe>__<hash>") as a fallback

    Why multiple keys?
      - Names may differ slightly (punctuation, case, sanitisation).
      - The short hash is stable and unambiguous.
    """
    index: Dict[str, Dict[str, Any]] = {}
    if not meta_dir.exists():
        return index

    for mf in meta_dir.glob("*.json"):
        try:
            with mf.open("r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            continue

        keys: List[str] = []

        # Best key: recording hash
        rec = meta.get("recording") or {}
        sh = rec.get("audio_sha256_short")
        if isinstance(sh, str) and sh:
            keys.append(sh)

        # System names
        sys_ctx = (meta.get("context") or {}).get("system") or {}
        for k in (sys_ctx.get("name"), sys_ctx.get("name_sanitized")):
            if isinstance(k, str) and k:
                keys.append(k)

        # Filename stem fallback
        keys.append(mf.stem)

        for k in keys:
            if k and k not in index:
                index[k] = meta

    return index


def load_covariates(points: List[SystemPoint], meta_dir: Path) -> None:
    """
    Enrich SystemPoint objects with covariates from metadata.

    Best-effort approach:
      - If we find metadata: fill covariates.
      - If not: leave covariates at defaults (None/False/1).

    Matching strategy (in order):
      1) short_hash (recording.audio_sha256_short)
      2) base (stem) "<SystemSafe>__<hash>"
      3) system name (context.system.name)
    """
    index = build_metadata_index(meta_dir)
    if not index:
        return

    for p in points:
        candidates: List[str] = []
        if isinstance(p.short_hash, str) and p.short_hash:
            candidates.append(p.short_hash)
        if isinstance(p.base, str) and p.base:
            candidates.append(p.base)
        if isinstance(p.name, str) and p.name:
            candidates.append(p.name)

        meta: Optional[Dict[str, Any]] = None
        for key in candidates:
            if key in index:
                meta = index[key]
                break
        if meta is None:
            continue

        stars = (meta.get("bodies") or {}).get("stars") or []
        if not isinstance(stars, list) or not stars:
            continue

        # Primary star = BodyID 0 if present, else first entry
        primary = None
        for s in stars:
            if isinstance(s, dict) and s.get("BodyID") == 0:
                primary = s
                break
        if primary is None:
            primary = stars[0] if isinstance(stars[0], dict) else None
        if primary is None:
            continue

        p.primary_star_type = primary.get("StarType") if isinstance(primary.get("StarType"), str) else None
        age = primary.get("Age_MY")
        p.primary_age_my = float(age) if isinstance(age, (int, float)) else None
        p.stars_count = len([s for s in stars if isinstance(s, dict)])
        p.has_tts = any(str(s.get("StarType", "")).upper().startswith("TTS") for s in stars if isinstance(s, dict))

        if p.primary_star_type:
            p.is_active = (_classify_star_type(p.primary_star_type) == "active")
        else:
            p.is_active = None


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def centroid(points: List[SystemPoint]) -> np.ndarray:
    """Centroid of system positions (used as a sensible initialisation)."""
    return np.mean(np.stack([p.pos for p in points]), axis=0)


def residuals_summary(points: List[SystemPoint], log10_pred_r: np.ndarray) -> Dict[str, Any]:
    """
    Residual summary computed in log10(r) space.

    We keep everything in log10(r) during fitting because:
      - intensities span orders of magnitude
      - errors should be relative, not absolute
    """
    log_obs = np.array([p.log10_r for p in points], dtype=np.float64)
    res = log_obs - log10_pred_r
    per = []
    for p, lp, r in zip(points, log10_pred_r.tolist(), res.tolist()):
        per.append({
            "name": p.name,
            "snr_db_obs": p.snr_db,
            # convert predicted log10(r) back to "dB energy ratio" for readability:
            "snr_db_pred": 10.0 * lp,
            "residual_log10r": float(r),
        })
    return {
        "per_system": per,
        "rmse_log10r": float(np.sqrt(np.mean(res ** 2))),
        "mae_log10r": float(np.mean(np.abs(res))),
        "max_abs_residual_log10r": float(np.max(np.abs(res))),
    }


# ---------------------------------------------------------------------------
# Shared modelling helpers (K elimination + constraints)
# ---------------------------------------------------------------------------

def _penalty_if_out_of_bounds(x: float, lo: float, hi: float, scale: float = 1e6) -> float:
    """
    Soft penalty function to impose bounds in Nelder-Mead (which doesn't support bounds).

    If x is within [lo, hi] -> 0
    If outside -> quadratic penalty
    """
    if lo <= x <= hi:
        return 0.0
    if x < lo:
        d = lo - x
    else:
        d = x - hi
    return scale * (d ** 2)


def _analytical_logK_for_A(log_r: np.ndarray, alpha: float, dists: np.ndarray) -> float:
    """
    For Model A, given S and alpha, the best log10(K) is the median of:
        log10(K) = median( log10(r_i) + alpha*log10(d_i) )
    """
    return float(np.median(log_r + alpha * np.log10(np.maximum(dists, 1.0))))


def _analytical_logK_for_B(log_r: np.ndarray, alpha: float, log10_g: np.ndarray, dists: np.ndarray) -> float:
    """
    For Model B, given S, alpha and per-system log10(g_i), the best log10(K) is:
        log10(K) = median( log10(r_i) - log10(g_i) + alpha*log10(d_i) )
    """
    return float(np.median(log_r - log10_g + alpha * np.log10(np.maximum(dists, 1.0))))


def _distance_regularizer(S: np.ndarray, c: np.ndarray, weight: float) -> float:
    """
    Optional regularizer to prevent the solution from drifting to extremely far coordinates.
    This is NOT a physics assumption; it's a numerical stabilizer when data is scarce.
    """
    if weight <= 0:
        return 0.0
    d = float(np.linalg.norm(S - c))
    return weight * (d ** 2)


# ---------------------------------------------------------------------------
# Model A — isotropic point source
# ---------------------------------------------------------------------------

def fit_model_A(
    points: List[SystemPoint],
    alpha_bounds: Tuple[float, float] = (0.5, 4.0),
    source_reg_weight: float = 0.0,
) -> Dict[str, Any]:
    """
    Model A in log10 space:
        log10(r_i) = log10(K) - alpha*log10(d_i)

    Unknowns: S=(sx,sy,sz), alpha, K
    Implementation details:
      - We eliminate K analytically (median), reducing the optimisation to 4D.
      - We fit in log10(r) space for stability.
      - Nelder-Mead is used (robust but unconstrained), so we add:
          * soft bounds on alpha
          * optional distance regularization for numerical stability
    """
    n = len(points)
    if n < 4:
        return {"error": f"Need at least 4 systems for Model A, got {n}"}

    X = np.stack([p.pos for p in points])
    log_r = np.array([p.log10_r for p in points], dtype=np.float64)

    c = centroid(points)

    def objective(params: np.ndarray) -> float:
        sx, sy, sz, log_alpha = params
        S = np.array([sx, sy, sz], dtype=np.float64)
        alpha = math.exp(log_alpha)

        # Penalties to keep alpha reasonable
        pen = _penalty_if_out_of_bounds(alpha, alpha_bounds[0], alpha_bounds[1], scale=1e6)
        pen += _distance_regularizer(S, c, weight=source_reg_weight)

        dists = np.linalg.norm(X - S, axis=1)
        if np.any(dists < 1.0):
            return 1e12 + pen

        logK = _analytical_logK_for_A(log_r, alpha, dists)
        log_pred = logK - alpha * np.log10(dists)
        sse = float(np.sum((log_r - log_pred) ** 2))
        return sse + pen

    cx, cy, cz = c.tolist()
    starts = [
        np.array([cx, cy, cz, math.log(2.0)], dtype=np.float64),
        np.array([cx + 500, cy, cz - 500, math.log(2.0)], dtype=np.float64),
        np.array([cx - 500, cy, cz + 500, math.log(1.5)], dtype=np.float64),
        np.array([cx, cy + 100, cz, math.log(3.0)], dtype=np.float64),
    ]

    best = None
    best_val = float("inf")
    for x0 in starts:
        try:
            res = minimize(
                objective,
                x0,
                method="Nelder-Mead",
                options={"maxiter": 80000, "xatol": 0.1, "fatol": 1e-7},
            )
            if res.fun < best_val:
                best_val = float(res.fun)
                best = res
        except Exception:
            continue

    if best is None:
        return {"error": "Optimization failed (no successful restart)"}

    sx, sy, sz, log_alpha = best.x
    S = np.array([sx, sy, sz], dtype=np.float64)
    alpha = float(math.exp(log_alpha))
    dists = np.linalg.norm(X - S, axis=1)
    logK = _analytical_logK_for_A(log_r, alpha, dists)

    # Predicted log10(r) (avoid 10** until after)
    log10_pred = (logK - alpha * np.log10(np.maximum(dists, 1.0))).astype(np.float64)

    return {
        "model": "A",
        "converged": bool(best.success),
        "source": [float(sx), float(sy), float(sz)],
        "alpha": float(alpha),
        "K_log10": float(logK),
        "objective_value": float(best_val),
        "dist_to_centroid_ly": float(np.linalg.norm(S - c)),
        "dist_nearest_system_ly": float(np.min(dists)),
        "dist_farthest_system_ly": float(np.max(dists)),
        "residuals": residuals_summary(points, log10_pred),
        "interpretation": _interpret_A(alpha, best_val, n),
        "params": {
            "alpha_bounds": [float(alpha_bounds[0]), float(alpha_bounds[1])],
            "source_reg_weight": float(source_reg_weight),
        },
    }


def _interpret_A(alpha: float, obj_val: float, n_systems: int) -> str:
    """
    Friendly interpretation string (not a proof; a quick diagnostic).
    """
    parts: List[str] = []
    avg = obj_val / max(n_systems, 1)
    if avg > 0.5:
        parts.append("Poor fit: isotropic point-source model likely inadequate.")
    else:
        parts.append("Fit may be acceptable for the isotropic model (check residuals/outliers).")

    if alpha < 0.7:
        parts.append("Alpha is very low: attenuation nearly flat (often suspicious).")
    elif 1.5 <= alpha <= 2.5:
        parts.append("Alpha near 2: consistent with 1/r² (3D isotropic), if the model is valid.")
    elif alpha > 3.5:
        parts.append("Alpha is very high: very steep attenuation (often a sign of model mismatch or sparse data).")
    else:
        parts.append(f"Alpha={alpha:.2f}: non-standard attenuation exponent.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Model B — source + per-system gain (regularized)
# ---------------------------------------------------------------------------

def fit_model_B(
    points: List[SystemPoint],
    lambda_reg: float = 1.0,
    alpha_bounds: Tuple[float, float] = (0.5, 4.0),
    source_reg_weight: float = 0.0,
) -> Dict[str, Any]:
    """
    Model B in log10 space:
        log10(r_i) = log10(g_i) + log10(K) - alpha*log10(d_i)

    Here, log10(g_i) is a free parameter per system, BUT we penalize it so it doesn't
    trivially absorb everything:
        objective = SSE + lambda_reg * sum(log10(g_i)^2)

    We also eliminate K analytically (median) for stability.
    """
    n = len(points)
    if n < 4:
        return {"error": f"Need at least 4 systems for Model B, got {n}"}

    X = np.stack([p.pos for p in points])
    log_r = np.array([p.log10_r for p in points], dtype=np.float64)
    c = centroid(points)

    # params = [sx, sy, sz, log_alpha, log10_g_0..log10_g_{n-1}]
    def objective(params: np.ndarray) -> float:
        sx, sy, sz = params[0], params[1], params[2]
        S = np.array([sx, sy, sz], dtype=np.float64)
        alpha = math.exp(params[3])
        log10_g = params[4:4 + n]

        pen = _penalty_if_out_of_bounds(alpha, alpha_bounds[0], alpha_bounds[1], scale=1e6)
        pen += _distance_regularizer(S, c, weight=source_reg_weight)

        dists = np.linalg.norm(X - S, axis=1)
        if np.any(dists < 1.0):
            return 1e12 + pen

        logK = _analytical_logK_for_B(log_r, alpha, log10_g, dists)
        log_pred = log10_g + logK - alpha * np.log10(dists)
        sse = float(np.sum((log_r - log_pred) ** 2))
        reg = float(lambda_reg * np.sum(log10_g ** 2))
        return sse + reg + pen

    cx, cy, cz = c.tolist()
    x0 = np.concatenate([
        np.array([cx, cy, cz, math.log(2.0)], dtype=np.float64),
        np.zeros(n, dtype=np.float64),
    ])

    try:
        res = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 150000, "xatol": 0.1, "fatol": 1e-7},
        )
    except Exception as e:
        return {"error": str(e)}

    sx, sy, sz = res.x[0], res.x[1], res.x[2]
    S = np.array([sx, sy, sz], dtype=np.float64)
    alpha = float(math.exp(res.x[3]))
    log10_g = res.x[4:4 + n]
    dists = np.linalg.norm(X - S, axis=1)
    logK = _analytical_logK_for_B(log_r, alpha, log10_g, dists)

    log10_pred = (log10_g + logK - alpha * np.log10(np.maximum(dists, 1.0))).astype(np.float64)

    # Objective decomposition for transparency:
    sse = float(np.sum((log_r - log10_pred) ** 2))
    reg = float(lambda_reg * np.sum(log10_g ** 2))
    total = float(res.fun)

    gains_info = []
    for i, p in enumerate(points):
        g_lin = float(10.0 ** log10_g[i])
        gains_info.append({
            "name": p.name,
            "log10_g": float(log10_g[i]),
            "g_linear": g_lin,
            "g_db": float(10.0 * math.log10(max(g_lin, 1e-12))),
        })

    return {
        "model": "B",
        "lambda_reg": float(lambda_reg),
        "converged": bool(res.success),
        "source": [float(sx), float(sy), float(sz)],
        "alpha": float(alpha),
        "K_log10": float(logK),
        "objective_value": total,
        "objective_decomposition": {
            "fit_error_sse": sse,
            "regularization": reg,
            "penalty_fraction": reg / max(total, 1e-12),
            "note": "Interpretation: if penalty_fraction is high, fitted gains are large (and heavily penalized) → per-system/environment effects dominate. If penalty_fraction is low, gains stay near 1 → the spatial model explains most variance.",
        },
        "dist_to_centroid_ly": float(np.linalg.norm(S - c)),
        "dist_nearest_system_ly": float(np.min(dists)),
        "dist_farthest_system_ly": float(np.max(dists)),
        "gains": gains_info,
        "residuals": residuals_summary(points, log10_pred),
        "interpretation": _interpret_B(gains_info),
        "params": {
            "alpha_bounds": [float(alpha_bounds[0]), float(alpha_bounds[1])],
            "source_reg_weight": float(source_reg_weight),
        },
    }


def _interpret_B(gains: List[Dict[str, Any]]) -> str:
    """
    Quickly highlight systems requiring large gain offsets.
    """
    high = [g for g in gains if abs(float(g["g_db"])) > 5.0]
    if not high:
        return "All per-system gains within ±5 dB: spatial model explains most variance (with small scene scaling)."
    names = ", ".join(g["name"] for g in high)
    return (
        f"Large gain offsets (>5 dB) for: {names}. "
        "These systems deviate strongly from the spatial model — likely 'audio scene' / environment effect."
    )


# ---------------------------------------------------------------------------
# Model C — source + covariates (interpretable gain model)
# ---------------------------------------------------------------------------

def fit_model_C(
    points: List[SystemPoint],
    alpha_bounds: Tuple[float, float] = (0.5, 4.0),
    source_reg_weight: float = 0.0,
) -> Dict[str, Any]:
    """
    Model C replaces free gains (Model B) with covariates:

        log10(g_i) = beta0
                    + beta_age   * log10(Age_MY_i)
                    + beta_stars * stars_count_i
                    + beta_tts   * has_tts_i
                    + beta_act   * is_active_i

    Then:
        log10(r_i) = log10(g_i) + log10(K) - alpha*log10(d_i)

    Notes:
      - Everything stays in log10 space (no exp / natural logs).
      - K is eliminated analytically (median), as in Models A and B.
      - This model is only meaningful if metadata covariates are present for enough systems.
    """
    n = len(points)
    if n < 4:
        return {"error": f"Need at least 4 systems for Model C, got {n}"}

    # Require at least a couple of systems with Age_MY, otherwise it's too underconstrained.
    n_with_age = sum(1 for p in points if p.primary_age_my is not None)
    if n_with_age < 2:
        return {
            "model": "C",
            "skipped": True,
            "reason": f"Only {n_with_age}/{n} systems have Age_MY in metadata. "
                      "Populate bodies.stars in metadata or collect more data.",
        }

    X = np.stack([p.pos for p in points])
    log_r = np.array([p.log10_r for p in points], dtype=np.float64)
    c = centroid(points)

    # Build covariate matrix
    ages = [p.primary_age_my for p in points if p.primary_age_my is not None]
    median_age = float(np.median(ages)) if ages else 1000.0

    def cov_row(p: SystemPoint) -> np.ndarray:
        age = p.primary_age_my if p.primary_age_my is not None else median_age
        # is_active is a best-effort covariate; it can be None when metadata is missing.
        # Default policy: treat "unknown" as 0.0 (same as calm/False) so we do NOT invent signal.
        # If you want neutrality, add a separate "unknown" indicator covariate instead.
        active = 1.0 if (p.is_active is True) else 0.0
        return np.array([
            math.log10(max(age, 1.0)),
            float(p.stars_count),
            1.0 if p.has_tts else 0.0,
            active,
        ], dtype=np.float64)

    COV = np.stack([cov_row(p) for p in points])  # shape (n, 4)

    # params = [sx, sy, sz, log_alpha, beta0, beta_age, beta_stars, beta_tts, beta_active]
    def objective(params: np.ndarray) -> float:
        sx, sy, sz = params[0], params[1], params[2]
        S = np.array([sx, sy, sz], dtype=np.float64)
        alpha = math.exp(params[3])
        betas = params[4:9]  # 5 betas total (intercept + 4 covariates)

        pen = _penalty_if_out_of_bounds(alpha, alpha_bounds[0], alpha_bounds[1], scale=1e6)
        pen += _distance_regularizer(S, c, weight=source_reg_weight)

        dists = np.linalg.norm(X - S, axis=1)
        if np.any(dists < 1.0):
            return 1e12 + pen

        log10_g = betas[0] + COV @ betas[1:]
        logK = float(np.median(log_r - log10_g + alpha * np.log10(np.maximum(dists, 1.0))))
        log_pred = log10_g + logK - alpha * np.log10(dists)
        sse = float(np.sum((log_r - log_pred) ** 2))
        return sse + pen

    cx, cy, cz = c.tolist()
    x0 = np.array([cx, cy, cz, math.log(2.0), 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    try:
        res = minimize(
            objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 150000, "xatol": 0.1, "fatol": 1e-7},
        )
    except Exception as e:
        return {"model": "C", "error": str(e)}

    sx, sy, sz = res.x[0], res.x[1], res.x[2]
    S = np.array([sx, sy, sz], dtype=np.float64)
    alpha = float(math.exp(res.x[3]))
    betas = res.x[4:9]

    dists = np.linalg.norm(X - S, axis=1)
    log10_g = betas[0] + COV @ betas[1:]
    logK = float(np.median(log_r - log10_g + alpha * np.log10(np.maximum(dists, 1.0))))

    log10_pred = (log10_g + logK - alpha * np.log10(np.maximum(dists, 1.0))).astype(np.float64)

    cov_effects = []
    for i, p in enumerate(points):
        cov_effects.append({
            "name": p.name,
            "primary_star_type": p.primary_star_type,
            "primary_age_my": p.primary_age_my,
            "stars_count": p.stars_count,
            "has_tts": p.has_tts,
            "is_active": p.is_active,
            "log10_g_from_covariates": float(log10_g[i]),
            "g_db": float(10.0 * log10_g[i]),
        })

    return {
        "model": "C",
        "converged": bool(res.success),
        "source": [float(sx), float(sy), float(sz)],
        "alpha": float(alpha),
        "K_log10": float(logK),
        "objective_value": float(res.fun),
        "dist_to_centroid_ly": float(np.linalg.norm(S - c)),
        "dist_nearest_system_ly": float(np.min(dists)),
        "dist_farthest_system_ly": float(np.max(dists)),
        "betas": {
            "beta0_intercept": float(betas[0]),
            "beta_age_log10": float(betas[1]),
            "beta_stars_count": float(betas[2]),
            "beta_has_tts": float(betas[3]),
            "beta_is_active": float(betas[4]),
        },
        "covariate_effects": cov_effects,
        "residuals": residuals_summary(points, log10_pred),
        "interpretation": _interpret_C(betas),
        "params": {
            "alpha_bounds": [float(alpha_bounds[0]), float(alpha_bounds[1])],
            "source_reg_weight": float(source_reg_weight),
        },
    }


def _interpret_C(betas: np.ndarray) -> str:
    """
    Quick interpretation of covariate coefficients.

    Reminder: coefficients operate on log10(g).
      - A +0.3 beta roughly corresponds to +3 dB shift, per +1 unit of covariate.
    """
    beta_age = float(betas[1])
    beta_stars = float(betas[2])
    beta_tts = float(betas[3])
    beta_act = float(betas[4])

    parts: List[str] = []
    if beta_age > 0.1:
        parts.append("Older stars → higher gain (higher SNR): consistent with calmer environments.")
    elif beta_age < -0.1:
        parts.append("Younger stars → higher gain: unexpected; check data/assumptions.")

    if abs(beta_stars) > 0.1:
        parts.append(("More companion stars reduces gain." if beta_stars < 0 else "More companion stars increases gain."))

    if beta_tts < -0.2:
        parts.append("TTS presence reduces gain significantly: supports 'noisy environment reduces SNR'.")
    elif beta_tts > 0.2:
        parts.append("TTS presence increases gain: unexpected.")

    if abs(beta_act) > 0.1:
        parts.append(("Active star types reduce gain." if beta_act < 0 else "Active star types increase gain."))

    if not parts:
        parts.append("Covariate effects are small; environment may not be the main driver, or data is insufficient.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Bootstrap / stability analysis
# ---------------------------------------------------------------------------

def _extract_source(fit_result: Dict[str, Any]) -> Optional[np.ndarray]:
    s = fit_result.get("source")
    if isinstance(s, list) and len(s) == 3:
        try:
            return np.array(s, dtype=np.float64)
        except Exception:
            return None
    return None


def bootstrap_fit(
    points: List[SystemPoint],
    model: str,
    lambda_b: float,
    alpha_bounds: Tuple[float, float],
    source_reg_weight: float,
    n_bootstrap: int,
    rng_seed: int,
) -> Dict[str, Any]:
    """
    Stability checks.

    1) Leave-one-out (LOO): remove one system and refit.
       Requires n >= 5 (so remaining >= 4).

    2) Resampling with replacement: sample N systems with replacement and refit.
       With small N, duplicates can make the fit degenerate; we deduplicate by name.

    Outputs:
      - per-fit sources
      - robust spread statistics in ly
    """
    n = len(points)
    rng = random.Random(rng_seed)

    def fit_fn(subset: List[SystemPoint]) -> Dict[str, Any]:
        if model == "A":
            return fit_model_A(subset, alpha_bounds=alpha_bounds, source_reg_weight=source_reg_weight)
        if model == "B":
            return fit_model_B(subset, lambda_reg=lambda_b, alpha_bounds=alpha_bounds, source_reg_weight=source_reg_weight)
        return fit_model_C(subset, alpha_bounds=alpha_bounds, source_reg_weight=source_reg_weight)

    loo_results: List[Dict[str, Any]] = []
    if n >= 5:
        for i in range(n):
            subset = [p for j, p in enumerate(points) if j != i]
            try:
                r = fit_fn(subset)
                s = _extract_source(r)
                if s is not None and "error" not in r and not r.get("skipped"):
                    loo_results.append({
                        "removed_system": points[i].name,
                        "source": s.tolist(),
                        "alpha": r.get("alpha"),
                        "objective_value": r.get("objective_value"),
                        "converged": r.get("converged", False),
                    })
                else:
                    loo_results.append({
                        "removed_system": points[i].name,
                        "note": "fit failed or skipped",
                        "raw": r,
                    })
            except Exception as e:
                loo_results.append({"removed_system": points[i].name, "error": str(e)})
    else:
        loo_results.append({"note": f"LOO skipped: only {n} systems (need >= 5)"})

    # Resampling
    resample_attempts = n_bootstrap if n >= 8 else min(n_bootstrap, 50)
    resample_results: List[Dict[str, Any]] = []
    skipped_too_few_unique = 0
    skipped_fit_failed = 0

    for _ in range(resample_attempts):
        sample = [rng.choice(points) for _ in range(n)]
        # Deduplicate (by name) to avoid pathologically repeated samples
        seen = set()
        unique: List[SystemPoint] = []
        for p in sample:
            if p.name not in seen:
                seen.add(p.name)
                unique.append(p)

        if len(unique) < 4:
            skipped_too_few_unique += 1
            continue

        try:
            r = fit_fn(unique)
            s = _extract_source(r)
            if s is not None and "error" not in r and not r.get("skipped"):
                resample_results.append({
                    "n_unique": len(unique),
                    "source": s.tolist(),
                    "alpha": r.get("alpha"),
                    "objective_value": r.get("objective_value"),
                })
            else:
                skipped_fit_failed += 1
        except Exception:
            skipped_fit_failed += 1
            continue

    def source_stats(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        sources = [np.array(r["source"], dtype=np.float64) for r in results if "source" in r]
        if len(sources) < 2:
            return None
        arr = np.stack(sources, axis=0)
        std = np.std(arr, axis=0)
        return {
            "n_fits": int(arr.shape[0]),
            "median_source": np.median(arr, axis=0).tolist(),
            "p10_source": np.percentile(arr, 10, axis=0).tolist(),
            "p90_source": np.percentile(arr, 90, axis=0).tolist(),
            "std_source": std.tolist(),
            "spread_ly": float(np.mean(std)),
        }

    loo_stats = source_stats(loo_results)
    resample_stats = source_stats(resample_results)

    # Simple stability assessment (rule-of-thumb, not a guarantee)
    stability = "unknown"
    if loo_stats:
        spread = loo_stats["spread_ly"]
        if spread < 500:
            stability = "stable (LOO spread < 500 ly)"
        elif spread < 1500:
            stability = "moderately stable (LOO spread 500–1500 ly)"
        else:
            stability = f"unstable (LOO spread ~{spread:.0f} ly) — likely insufficient data or wrong model"

    return {
        "model": model,
        "n_systems": n,
        "stability_assessment": stability,
        "loo": {
            "results": loo_results,
            "stats": loo_stats,
        },
        "resample": {
            "attempts": resample_attempts,
            "fits_ok": len(resample_results),
            "skipped_too_few_unique": skipped_too_few_unique,
            "skipped_fit_failed_or_skipped": skipped_fit_failed,
            "stats": resample_stats,
        },
    }


# ---------------------------------------------------------------------------
# CLI and orchestration
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit a hypothetical source from analysis/summary.json and system coordinates."
    )
    p.add_argument("--summary", default="analysis/summary.json", help="Path to analysis/summary.json")
    p.add_argument("--metadata", default="metadata", help="Metadata folder (for Model C covariates)")
    p.add_argument("--out", default="analysis", help="Output folder (default: analysis)")
    p.add_argument("--model", choices=["A", "B", "C", "all"], default="all", help="Which model(s) to run")
    p.add_argument("--lambda-b", type=float, default=1.0, help="Regularization weight for Model B gains")
    p.add_argument("--no-bootstrap", action="store_true", help="Skip bootstrap analysis")
    p.add_argument("--bootstrap-n", type=int, default=200, help="Bootstrap iterations (resampling)")
    p.add_argument("--min-systems", type=int, default=4, help="Minimum systems to attempt a fit")
    p.add_argument("--alpha-min", type=float, default=0.5, help="Min alpha bound (soft)")
    p.add_argument("--alpha-max", type=float, default=4.0, help="Max alpha bound (soft)")
    p.add_argument("--source-reg", type=float, default=0.0,
                   help="Optional stabilizer: penalize distance of S from centroid (0 disables).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap sampling")
    return p.parse_args()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()

    summary_path = (repo_root / args.summary).resolve()
    meta_dir = (repo_root / args.metadata).resolve()
    out_dir = (repo_root / args.out).resolve()

    if not summary_path.exists():
        print(f"[ERROR] summary.json not found: {summary_path}")
        print("        Run tools/analyze_clicks.py first.")
        return 2

    points = load_summary(summary_path)
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
    for p in points:
        st = p.primary_star_type or "?"
        age = f"{int(p.primary_age_my)}" if p.primary_age_my is not None else "?"
        active = "active" if p.is_active else ("calm" if p.is_active is not None else "?")
        sh = p.short_hash or "?"
        print(
            f"  {p.name:40s} Ii={p.snr_db:6.2f} dB  pos=[{p.pos[0]:.0f},{p.pos[1]:.0f},{p.pos[2]:.0f}]"
            f"  StarType={st:8s} Age_MY={age:>6s}  active={active:5s}  TTS={'yes' if p.has_tts else 'no'}  hash={sh}"
        )

    alpha_bounds = (float(args.alpha_min), float(args.alpha_max))

    models = ["A", "B", "C"] if args.model == "all" else [args.model]

    fit_results: Dict[str, Any] = {
        "n_systems": len(points),
        "systems_used": [p.name for p in points],
        "alpha_bounds": [alpha_bounds[0], alpha_bounds[1]],
        "source_reg_weight": float(args.source_reg),
        "models": {},
    }
    bootstrap_results: Dict[str, Any] = {
        "n_systems": len(points),
        "models": {},
    }

    for m in models:
        print(f"\n--- Model {m} ---")

        if m == "A":
            result = fit_model_A(points, alpha_bounds=alpha_bounds, source_reg_weight=args.source_reg)
        elif m == "B":
            result = fit_model_B(points, lambda_reg=args.lambda_b, alpha_bounds=alpha_bounds, source_reg_weight=args.source_reg)
        else:
            result = fit_model_C(points, alpha_bounds=alpha_bounds, source_reg_weight=args.source_reg)

        fit_results["models"][m] = result

        if "error" in result:
            print(f"[WARN] Model {m}: {result['error']}")
            continue
        if result.get("skipped"):
            print(f"[INFO] Model {m} skipped: {result.get('reason')}")
            continue

        src = result.get("source") or [None, None, None]
        alpha = result.get("alpha")
        rmse = (result.get("residuals") or {}).get("rmse_log10r")
        print(f"  Source : [{src[0]:.0f}, {src[1]:.0f}, {src[2]:.0f}]")
        print(f"  Alpha  : {alpha:.3f}" if isinstance(alpha, (int, float)) else "  Alpha  : n/a")
        print(f"  RMSE   : {rmse:.4f} log10(r)" if isinstance(rmse, (int, float)) else "  RMSE   : n/a")
        print(f"  Note   : {result.get('interpretation', '')}")

        if not args.no_bootstrap:
            print(f"  Bootstrap... (n={args.bootstrap_n})")
            bs = bootstrap_fit(
                points=points,
                model=m,
                lambda_b=float(args.lambda_b),
                alpha_bounds=alpha_bounds,
                source_reg_weight=float(args.source_reg),
                n_bootstrap=int(args.bootstrap_n),
                rng_seed=int(args.seed),
            )
            bootstrap_results["models"][m] = bs
            print(f"  Stability: {bs.get('stability_assessment')}")

    # Write outputs
    out_fit = out_dir / "source_fit.json"
    write_json(out_fit, fit_results)
    print(f"\nFit results written to: {out_fit}")

    if not args.no_bootstrap and bootstrap_results["models"]:
        out_bs = out_dir / "source_bootstrap.json"
        write_json(out_bs, bootstrap_results)
        print(f"Bootstrap results written to: {out_bs}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
