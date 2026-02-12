# click_route_plan.py

## Overview

`click_route_plan.py` generates an exploration route for discovering
potential new *click systems* in Elite Dangerous.

It is designed as an operational mission tool, not an analysis tool.\
Each pilot runs it locally using their own Journal logs.

The script:

-   Uses the canonical list of known click systems (positives) from the
    repository
-   Uses the pilot's local Journal logs to determine:
    -   Systems already visited (negatives)
    -   The last visited system (route starting point)
-   Generates two exploration strategies:
    -   FRONTIÈRES (cluster boundary exploration)
    -   SPOKES (axis-based radial exploration)
-   Resolves candidate 3D coordinates into real jumpable system names
    via EDSM (with optional Spansh attempt first)
-   Outputs a single file: route_plan.md

No intermediate CSV files are generated.

------------------------------------------------------------------------

## Conceptual Model

The route planner operates under the geographical hypothesis (H1):

> Click systems may exhibit non-random spatial structure.

To test this, it generates two types of routes:

### FRONTIÈRES

-   Detects clusters of known click systems (DBSCAN)
-   Allocates 15 targets proportionally to cluster sizes
-   Generates points on a shell around each cluster
-   Resolves those points to real systems
-   Produces a single optimized route from the pilot's current location

Purpose: test cluster boundaries.

------------------------------------------------------------------------

### SPOKES

-   Computes global barycenter of click systems
-   Computes PCA axes
-   Generates radial exploration targets along principal directions
-   Resolves to real systems
-   Produces a single optimized route

Purpose: test anisotropy or filament-like structure.

------------------------------------------------------------------------

## Key Rules

-   Starting system = last known system in the pilot's Journal logs
-   Excluded targets:
    -   Known click systems
    -   Systems already visited by this pilot
-   Resolution is mandatory (point → real system name)
-   If resolution fails, radius expands progressively
-   If needed, jitter attempts are applied
-   Exactly 15 systems per approach (best effort)

------------------------------------------------------------------------

## Requirements

-   Python 3.10+
-   numpy
-   scikit-learn
-   requests

------------------------------------------------------------------------

## Usage

Example:

python tools\click_route_plan.py --logs "C:\Users\<you>\Saved Games\Frontier Developments\Elite Dangerous" --clicks analysis\summary.json --out route_plan.md

### Options

-   --resolver edsm_only\
    Use only EDSM (recommended if Spansh API is unreliable)

-   --seed `<int>`{=html}\
    Set random seed for reproducibility

------------------------------------------------------------------------

## Supported Click List Formats

-   JSON list of names
-   JSON list of objects containing system + coords
-   JSON dictionary mapping system → metadata
-   TXT file (one system per line)

Coordinates are required for spatial computation.

------------------------------------------------------------------------

## Output

route_plan.md

Example:

Start system: Pleiades Sector AB-W b2-4

FRONTIÈRES (15): Pleiades Sector AB-W b2-4 -\> Target1 -\> Target2 -\>
...

SPOKES (15): Pleiades Sector AB-W b2-4 -\> TargetA -\> TargetB -\> ...

Only names are listed. No distances are displayed.

------------------------------------------------------------------------

## Multi-Pilot Usage

This script is pilot-local:

-   Each pilot runs it with their own Journal logs
-   Only global positives are shared
-   Negatives are pilot-specific
-   The route file should not be committed to the repository

The repository stores: - Audio - Ingestion metadata - Aggregated
analysis results

The route planner is a mission tool, not a data artifact.

------------------------------------------------------------------------

## Design Philosophy

-   Operational simplicity
-   No global state mutation
-   No dependency on shared negative history
-   Reproducible routes via seed control
-   Clear separation between collection tools and analysis tools
