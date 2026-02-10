# fit_source.py

This script attempts to **fit a hypothetical source** responsible for the FSS click intensities
measured across multiple star systems in *Elite Dangerous*.

It is designed to be used **after** `tools/analyze_clicks.py` and relies on the same repository
layout.

---

## Purpose

Given:
- a set of star systems with known 3D coordinates,
- a reproducible audio click pattern,
- a per-system intensity metric (tick energy vs local background),

the script explores whether the observed intensity variations can be explained by:
- pure geometric attenuation,
- geometric attenuation plus per-system audio/environment scaling,
- geometric attenuation plus interpretable astrophysical/environmental covariates.

The script is **diagnostic**, not assertive:
it will always find a numerical optimum, but stability and residuals determine whether the
model is meaningful.

---

## Inputs

### Required
- `analysis/summary.json`  
  Produced by `tools/analyze_clicks.py`.
  Contains, for each system:
  - `star_pos` (x, y, z in light-years),
  - `Ii_db_median` (median tick/background energy ratio in dB),
  - identifiers (`base`, `audio_file`, hash).

### Optional
- `metadata/*.json`  
  Used only for **Model C**.
  Supplies stellar/environment covariates:
  - primary star type,
  - age (`Age_MY`),
  - number of stars,
  - presence of T Tauri stars (TTS).

If metadata is missing or incomplete, Model C is skipped gracefully.

---

## Models

### Model A - Isotropic point source (baseline)

Assumes a single source emitting isotropically:

```
r_i = K / d_i^alpha
```

where:
- `r_i` is the linear intensity ratio,
- `d_i` is the distance from source to system *i*,
- `alpha` is the attenuation exponent.

Characteristics:
- minimal assumptions,
- very sensitive to outliers,
- useful mainly as a falsification baseline.

---

### Model B - Source + per-system gain (regularized)

Extends Model A with a per-system multiplicative gain:

```
r_i = g_i * K / d_i^alpha
```

with a penalty on `g_i` to keep gains near 1.

Interpretation:
- allows each system to rescale intensity due to audio scene or environment,
- reveals *which* systems strongly deviate from spatial expectations.

**Penalty fraction**
- high → gains are large and actively penalized → environment dominates,
- low  → gains remain small → spatial model explains most variance.

---

### Model C - Source + covariates (interpretable)

Replaces free gains with a linear model:

```
log10(g_i) =
    beta0
  + beta_age   * log10(Age_MY)
  + beta_stars * stars_count
  + beta_tts   * has_tts
  + beta_act   * is_active
```

Notes:
- everything is computed in log10 space,
- `is_active` is **best-effort**:
  - `True` → 1,
  - `False` → 0,
  - `None`  → 0 (unknown treated conservatively; no signal invented).

This model is the most interpretable but requires sufficient metadata coverage.

---

## Numerical details

- All fitting is done in **log10(r)** space (stable across orders of magnitude).
- `K` is eliminated analytically (median-based) to reduce dimensionality.
- Optimisation uses **Nelder–Mead** with multiple restarts.
- Soft bounds are applied to `alpha` via penalties (no hard clipping).

---

## Output

### `analysis/source_fit.json`

For each model:
- fitted source position `[x, y, z]`,
- attenuation exponent `alpha`,
- residual statistics,
- model-specific diagnostics (gains, betas, penalty interpretation).

### `analysis/source_bootstrap.json` (if enabled)

Stability analysis using:
- leave-one-out refits,
- resampling with replacement.

Reports spatial spread of the fitted source.

---


## Usage

Basic (run all models + bootstrap):

```bash
python tools/fit_source.py
```

Run a single model:

```bash
python tools/fit_source.py --model A
python tools/fit_source.py --model B
python tools/fit_source.py --model C
```

Tuning Model B regularization (how hard we pull gains back toward 1):

```bash
python tools/fit_source.py --model B --lambda-b 2.0
```

Skip bootstrap (faster iteration while you’re debugging data/metadata joins):

```bash
python tools/fit_source.py --no-bootstrap
```

Stabilize if the fitted source “runs away” (small datasets can do that).
This adds a weak quadratic pull toward the centroid of your systems:

```bash
python tools/fit_source.py --source-reg 1e-8
```

Other useful knobs:

```bash
python tools/fit_source.py --alpha-min 0.5 --alpha-max 4.0
python tools/fit_source.py --bootstrap-n 300
python tools/fit_source.py --min-systems 6
```

---

## How to read the results

### Source coordinates
Each model returns a `"source": [x, y, z]` position in light-years,
in Elite Dangerous galactic coordinates.

**These coordinates are only meaningful if the bootstrap is stable.**
With few systems, the optimiser will always find a position, but it may
be physically worthless. Use `dist_nearest_system_ly` as a sanity check:
if the fitted source is tens of thousands of ly from all your systems, the
model is likely underconstrained.

Bootstrap stability (LOO spread):

| Spread       | Interpretation                        |
|--------------|---------------------------------------|
| < 500 ly     | Stable - position is usable           |
| 500-1500 ly  | Moderately stable                     |
| > 1500 ly    | Insufficient data - collect more      |

In practice, 15-20 systems with good galactic coverage are needed before
the fit starts to converge.

### Other signals
- **Unstable source position** -> geometry alone is insufficient.
- **Large gains or penalty fraction (Model B)** -> environment dominates.
- **Coherent covariate signs (Model C)** -> systematic astrophysical influence.
- **Model A failing but B/C stable** -> spatial signal exists but is modulated.

A failed model is not a dead end; it is information.
---

## Status

This script is intended as a **research and exploration tool**.
It is deliberately explicit, conservative, and heavily commented to
avoid accidental over-interpretation.
