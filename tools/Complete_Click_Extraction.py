"""
detect_clicks.py
================
Détecte la séquence de clics commune dans des fichiers FLAC.

Pattern attendu (~7.04s) :
  CLICK---CLICK---CLICK---CLICK·CLICK---CLICK---CLICK------CLICK---CLICK---CLICK·CLICK
  ←——— 3 espacés ———→←paire→←— 2 espacés —→←— pause —→←— 2 espacés —→←paire→

Usage :
  python detect_clicks.py fichier1.flac fichier2.flac ...
  python detect_clicks.py *.flac
  python detect_clicks.py dossier/
"""

import sys
import os
import numpy as np
import librosa
from scipy.signal import find_peaks, butter, filtfilt


# ── Paramètres calibrés sur les fichiers de référence ──────────────────────
EXPECTED_PERIOD     = 7.04    # secondes
PERIOD_TOLERANCE    = 0.5     # ±0.5s accepté
SHORT_IOI_MAX       = 0.20    # en dessous = paire rapprochée
LARGE_IOI_MIN       = 0.90    # au dessus = grande pause centrale
HP_CUTOFF           = 5000    # Hz, filtre passe-haut pour isoler les clics
ENERGY_PERCENTILE   = 99.5    # seuil de détection (percentile)
MIN_CLICK_DISTANCE  = 0.04    # secondes, distance minimale entre 2 clics
# ───────────────────────────────────────────────────────────────────────────


def load_and_detect_clicks(path: str) -> tuple[np.ndarray, float, int]:
    """Charge un FLAC et retourne les timestamps des clics détectés."""
    y, sr = librosa.load(path, sr=None)
    duration = len(y) / sr

    # Filtre passe-haut pour isoler l'énergie transitoire des clics
    b, a = butter(4, HP_CUTOFF / (sr / 2), btype='high')
    y_hf = filtfilt(b, a, y)

    # Énergie RMS sur fenêtres courtes
    hop = 64
    frame_len = 128
    energy = np.array([
        np.sqrt(np.mean(y_hf[i:i + frame_len] ** 2))
        for i in range(0, len(y_hf) - frame_len, hop)
    ])
    times = np.arange(len(energy)) * hop / sr

    threshold = np.percentile(energy, ENERGY_PERCENTILE)
    min_dist_frames = int(MIN_CLICK_DISTANCE * sr / hop)
    peaks, _ = find_peaks(energy, height=threshold, distance=min_dist_frames)

    return times[peaks], duration, sr


def ioi_labels(ioi: np.ndarray) -> str:
    """Convertit les IOI en labels S/M/L."""
    labels = []
    for x in ioi:
        if x < SHORT_IOI_MAX:
            labels.append('S')
        elif x < LARGE_IOI_MIN:
            labels.append('M')
        else:
            labels.append('L')
    return ''.join(labels)


def find_cycles(pt: np.ndarray, duration: float) -> dict:
    """
    Cherche les cycles dans les timestamps de clics.
    Utilise les paires rapprochées comme ancres.
    """
    ioi = np.diff(pt)
    pair_idx = np.where(ioi < SHORT_IOI_MAX)[0]

    if len(pair_idx) < 2:
        return {"error": "Pas assez de paires rapprochées détectées"}

    pair_times = np.array([(pt[i] + pt[i + 1]) / 2 for i in pair_idx])

    # Période = écart entre paires séparées par une autre paire
    periods = []
    for i in range(len(pair_times) - 2):
        p = pair_times[i + 2] - pair_times[i]
        if EXPECTED_PERIOD - PERIOD_TOLERANCE < p < EXPECTED_PERIOD + PERIOD_TOLERANCE:
            periods.append(p)

    if not periods:
        return {"error": f"Aucune période proche de {EXPECTED_PERIOD}s trouvée"}

    period = np.median(periods)

    # Extraire un cycle de référence
    # 11 clics, pattern attendu MMMSMMLMMS (2S, 1L)
    # Note: le span des 11 clics est ~5.76s (pas ~7.04s),
    # car le gap de 1.28s jusqu'au cycle suivant n'est pas inclus
    best_cycle = None
    for start_idx in range(len(pt) - 10):
        candidate = pt[start_idx:start_idx + 11]
        ioi_c = np.diff(candidate)
        lbl = ioi_labels(ioi_c)
        if lbl.count('S') == 2 and lbl.count('L') == 1:
            best_cycle = candidate
            break

    cycle_ioi = np.diff(best_cycle - best_cycle[0]) if best_cycle is not None else np.array([])

    # Validation du pattern attendu
    labels = ioi_labels(cycle_ioi)
    pattern_ok = labels.count('S') == 2 and labels.count('L') == 1

    n_expected = duration / period
    n_pairs = len(pair_idx)

    return {
        "period_s": round(period, 4),
        "n_cycles_expected": round(n_expected, 1),
        "n_pairs_detected": n_pairs,
        "clicks_per_cycle": len(best_cycle) if best_cycle is not None else None,
        "cycle_ioi": np.round(cycle_ioi, 3).tolist() if len(cycle_ioi) else [],
        "cycle_labels": labels,
        "pattern_valid": pattern_ok,
        "pair_times": np.round(pair_times, 3).tolist(),
    }


def format_cycle_ascii(ioi: list) -> str:
    """Représentation ASCII de la séquence de clics."""
    if not ioi:
        return "(aucun cycle extrait)"
    parts = ["CLICK"]
    for interval in ioi:
        if interval < SHORT_IOI_MAX:
            sep = "·"
        elif interval < LARGE_IOI_MIN:
            sep = "---"
        else:
            sep = "------"
        parts.append(sep)
        parts.append("CLICK")
    return "".join(parts)


def analyze_file(path: str) -> None:
    """Analyse complète d'un fichier FLAC."""
    name = os.path.basename(path)
    print(f"\n{'═' * 60}")
    print(f"📁 {name}")
    print(f"{'═' * 60}")

    try:
        pt, duration, sr = load_and_detect_clicks(path)
    except Exception as e:
        print(f"  ❌ Erreur de chargement : {e}")
        return

    print(f"  Durée         : {duration:.2f}s")
    print(f"  Sample rate   : {sr} Hz")
    print(f"  Clics bruts   : {len(pt)}")

    result = find_cycles(pt, duration)

    if "error" in result:
        print(f"  ❌ {result['error']}")
        return

    ok = "✅" if result["pattern_valid"] else "⚠️ "
    print(f"  Période cycle : {result['period_s']}s")
    print(f"  Cycles        : {result['n_cycles_expected']}")
    print(f"  Paires (·)    : {result['n_pairs_detected']}")
    print(f"  Clics/cycle   : {result['clicks_per_cycle']}")
    print(f"  Pattern       : {ok} {result['cycle_labels']}")
    print(f"\n  Séquence :")
    print(f"  {format_cycle_ascii(result['cycle_ioi'])}")
    print(f"\n  IOI détaillés : {result['cycle_ioi']}")
    print(f"  Légende       : · ≈ {[x for x in result['cycle_ioi'] if x < SHORT_IOI_MAX]}")
    print(f"                  --- ≈ {sorted(set([round(x,2) for x in result['cycle_ioi'] if SHORT_IOI_MAX <= x < LARGE_IOI_MIN]))}")
    print(f"                  ------ ≈ {[x for x in result['cycle_ioi'] if x >= LARGE_IOI_MIN]}")


def main():
    paths = []

    if len(sys.argv) < 2:
        print("Usage : python detect_clicks.py fichier1.flac [fichier2.flac ...] ou dossier/")
        sys.exit(1)

    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            for fn in sorted(os.listdir(arg)):
                if fn.lower().endswith('.flac'):
                    paths.append(os.path.join(arg, fn))
        elif os.path.isfile(arg):
            paths.append(arg)
        else:
            print(f"⚠️  Ignoré (introuvable) : {arg}")

    if not paths:
        print("Aucun fichier FLAC trouvé.")
        sys.exit(1)

    print(f"\n🔍 Analyse de {len(paths)} fichier(s) FLAC...")
    print(f"   Pattern attendu : 3 espacés + paire·paire + 2 espacés + pause + 2 espacés + paire·paire")
    print(f"   Période cible   : ~{EXPECTED_PERIOD}s")

    for path in paths:
        analyze_file(path)

    print(f"\n{'═' * 60}")
    print(f"✅ Analyse terminée — {len(paths)} fichier(s) traité(s)")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()