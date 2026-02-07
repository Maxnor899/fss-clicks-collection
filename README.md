# FSS Clicks Collection

## What is this repository?

This repository is a **community-driven data collection project** around the mysterious *FSS clicks* heard in Elite Dangerous.

The goal is to **collect clean, comparable data** so hypotheses can be tested on real evidence rather than impressions.

Each contribution consists of:

* a short audio recording (≤ 60 seconds)
* an automatically generated metadata file describing the system and context

---

## What are we trying to observe?

At this stage, one main observable is of interest:

1. **Click intensity** (how strongly the clicks stand out from background noise)

This observable may later be compared against system properties such as:

* number of stars
* number of planets
* system layout
* presence of biological or codex signals
* environmental subjects

No assumption is made about the cause of the clicks.

---

## Repository structure

```
fss-clicks-collection/
├─ audio/        # Final, renamed FLAC files
├─ metadata/     # Auto-generated JSON metadata
├─ incoming/     # Place ONE raw FLAC here before running the script
├─ tools/        # Python scripts
│  └─ ingest.py
├─ README.md
└─ .gitignore
```

---

## Contribution workflow (simple)

### 1. Record audio

* Record **up to 60 seconds** of audio while using the FSS (**containing the clicks....**)
* The file can be named **anything**
* Export as **FLAC** as described below : 

<img width="613" height="459" alt="image" src="https://github.com/user-attachments/assets/734190a0-3f04-4060-8902-855dd02dc8fc" />


Place the file into:

```
incoming/
```


⚠️ There must be **ONLY one FLAC file** in this folder.

---

### 2. Run the ingestion script

From the repo root:

```bash
python tools/ingest.py "System Name Here"
```

The script will:

* verify the audio duration
* compute a short SHA-256 hash of the file
* rename and move the audio into `audio/`
* extract all available system metadata from Elite Dangerous logs
* generate a matching JSON file in `metadata/`

Example output files:

```
audio/Praea_Eurl_RY-H_d10-0__a3f92c1d.flac
metadata/Praea_Eurl_RY-H_d10-0__a3f92c1d.json
```

---

### 3. Commit and open a PR

```bash
git add audio/ metadata/
git commit -m "Add FSS recording for Praea Eurl RY-H d10-0"
git push
```

Open a Pull Request.

Multiple recordings from the same system are expected and welcome.

---

## Naming and collision handling

File names always include a short hash derived from the audio content:

```
<SystemName>__<hash>.flac
<SystemName>__<hash>.json
```

This avoids conflicts when multiple CMDRs record the same system independently.

---

## About metadata

* Metadata is generated automatically from Elite Dangerous journal logs
* Fields that cannot be determined are set to `null`
* `null` means *not available in the logs*, **not** *does not exist*

Audio contains the signal itself.
Metadata provides the context.

---

## What this is NOT

* This is **not** a proof of a hidden signal
* This is **not** a claim that the clicks mean anything


It is a **data collection effort**, nothing more, nothing less.
Any future trilateration would depend entirely on what the data actually shows.

---

## Requirements

* Python 3.9+
* Elite Dangerous journal logs available locally
* FLAC recording (≤ 60s)
* mutagen (pip install mutagen)
---

## Final note

If the clicks turn out to be simple FSS ambience, this repository will help show it.
If they are not, this repository will help show that too.

Either outcome is a valid result.

Thanks for contributing.
