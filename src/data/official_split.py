# Skill source: external/scientific-agent-skills/SECURITY.md
# - Emphasises secure file handling, input validation, and robust error recovery.
# Skill source: external/AI-research-SKILLs/03-fine-tuning/unsloth/references/llms-full.md
# - General best practices for production data pipelines (logging, clarity).
# Skill source: external/AI-research-SKILLs/03-fine-tuning/unsloth/references/llms-txt.md
# - Reinforces clear, runnable Python examples with meaningful variable names.

"""
official_split.py — Verify ICBHI2017 Official 60/40 Patient-Wise Split
======================================================================

This script performs the following checks on the ICBHI2017 respiratory sound
database using the official train/test split file:

1. Reads ``data/official_split.txt`` (tab-separated: filename, train|test).
2. Extracts ``patient_id`` from each filename (first underscore-delimited field).
3. Groups all recordings by patient and verifies **no patient appears in both
   train and test sets** — detecting leakage if present.
4. Prints summary statistics: total files, train / test counts, and number of
   unique patients in each split.
5. Reads the per-recording ``.txt`` annotation files and computes the class
   distribution (crackles, wheezes, both, normal) for train and test.

Usage::

    python src/data/official_split.py

Environment variable ``ICBHI2017_ROOT`` may be set to the dataset root.
If not set, the script falls back to ``data/`` relative to the project root
(where this file lives at ``src/data/official_split.py``).
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, Tuple


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_data_root() -> Path:
    """Return the root directory containing the ICBHI2017 dataset.

    1. Honour the ``ICBHI2017_ROOT`` environment variable if set.
    2. Otherwise fall back to ``data/`` relative to the project root.
       The project root is two levels above this file:
       ``src/data/official_split.py`` → ``data/``.
    """
    env_root = os.environ.get("ICBHI2017_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if root.exists():
            return root
        print(f"[WARN] ICBHI2017_ROOT={env_root} does not exist; falling back.", file=sys.stderr)

    # Fallback: go up two directories from this script, then into 'data'
    script_dir = Path(__file__).resolve().parent  # src/data
    project_root = script_dir.parent.parent        # project root
    root = project_root / "data"
    return root


# ---------------------------------------------------------------------------
# Helper: parse the official split file
# ---------------------------------------------------------------------------

def parse_split_file(split_path: Path) -> Dict[str, str]:
    """Parse ``official_split.txt`` and return {filename: 'train'|'test'}.

    Expected format (tab-separated)::

        101_1b1_Al_sc_Meditron\ttest
        102_1b1_Ar_mc_Litt3200\ttrain

    Raises
    ------
    FileNotFoundError
        If *split_path* does not exist.
    ValueError
        If any line is malformed.
    """
    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    split_map: Dict[str, str] = {}
    with split_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(
                    f"Line {line_no} in {split_path} is malformed: {line!r}"
                )
            filename, subset = parts[0].strip(), parts[1].strip().lower()
            if subset not in ("train", "test"):
                raise ValueError(
                    f"Line {line_no}: unexpected subset {subset!r} (expected train/test)"
                )
            if filename in split_map:
                raise ValueError(
                    f"Line {line_no}: duplicate filename {filename!r}"
                )
            split_map[filename] = subset
    return split_map


# ---------------------------------------------------------------------------
# Helper: extract patient ID from filename
# ---------------------------------------------------------------------------

def extract_patient_id(filename: str) -> str:
    """Return the patient identifier from an ICBHI2017 recording name.

    Filename structure::

        {patient}_{recording_index}_{chest_location}_{acquisition_mode}_{equipment}

    The patient ID is the portion before the first underscore.
    Example: ``"101_1b1_Al_sc_Meditron"`` → ``"101"``.
    """
    return filename.split("_")[0]


# ---------------------------------------------------------------------------
# Helper: determine the label of a recording from its annotation file
# ---------------------------------------------------------------------------

def determine_label(annotation_path: Path) -> str:
    """Infer the lung-sound label from an ICBHI2017 annotation ``.txt`` file.

    Rules (applied row-by-row):
        - If **any** row contains ``crackles=1 AND wheezes=1`` → ``"both"``.
        - Else if any row contains ``crackles=1`` → ``"crackles"``.
        - Else if any row contains ``wheezes=1`` → ``"wheezes"``.
        - Otherwise → ``"normal"``.

    Returns
    -------
    str
        One of ``{"normal", "crackles", "wheezes", "both"}``.
    """
    has_crackles = False
    has_wheezes = False

    try:
        with annotation_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                cols = line.split("\t")
                if len(cols) < 4:
                    continue  # skip malformed rows silently
                # Columns: start_time, end_time, crackles(0/1), wheezes(0/1)
                crackles = int(cols[2])
                wheezes = int(cols[3])
                if crackles and wheezes:
                    return "both"  # early exit: strongest signal found
                if crackles:
                    has_crackles = True
                if wheezes:
                    has_wheezes = True
    except Exception as exc:
        print(f"[WARN] Could not read {annotation_path}: {exc}", file=sys.stderr)
        return "unknown"

    if has_crackles:
        return "crackles"
    if has_wheezes:
        return "wheezes"
    return "normal"


# ---------------------------------------------------------------------------
# Core verification logic
# ---------------------------------------------------------------------------

def verify_split(data_root: Path) -> None:
    """Run all split-verification checks and print the results."""
    split_path = data_root / "official_split.txt"
    audio_dir = data_root / "audio"

    print(f"Data root      : {data_root}")
    print(f"Split file     : {split_path}")
    print(f"Audio dir      : {audio_dir}")
    print()

    # 1. Parse split file
    split_map = parse_split_file(split_path)
    total_files = len(split_map)
    print(f"Total recordings in split file: {total_files}")

    # 2. Group by patient
    patient_to_files: Dict[str, list] = defaultdict(list)
    for filename in split_map:
        pid = extract_patient_id(filename)
        patient_to_files[pid].append(filename)

    unique_patients = len(patient_to_files)
    print(f"Unique patients              : {unique_patients}")

    # 3. Assign patients to train / test (check leakage)
    train_patients: Set[str] = set()
    test_patients: Set[str] = set()

    for pid, files in patient_to_files.items():
        subsets = {split_map[f] for f in files}
        if len(subsets) > 1:
            # LEAKAGE DETECTED — same patient in both train and test
            print(f"\n[LEAKAGE DETECTED] Patient {pid} appears in {subsets}!", file=sys.stderr)
            for fname in files:
                print(f"  {fname}: {split_map[fname]}", file=sys.stderr)
            raise SystemExit(1)

        if "train" in subsets:
            train_patients.add(pid)
        else:
            test_patients.add(pid)

    # Sanity: every patient should be in exactly one set
    assert len(train_patients) + len(test_patients) == unique_patients, \
        "Patient count mismatch after grouping!"

    n_train_files = sum(1 for s in split_map.values() if s == "train")
    n_test_files = total_files - n_train_files

    print(f"\nTrain recordings             : {n_train_files} ({n_train_files / total_files:.1%})")
    print(f"Test recordings              : {n_test_files} ({n_test_files / total_files:.1%})")
    print(f"Train patients               : {len(train_patients)}")
    print(f"Test patients                : {len(test_patients)}")
    print(f"Patient leakage?             : NO — clean split ✓")

    # 4. Compute per-file labels and class distribution
    print("\n--- Class distribution (from annotation files) ---")
    train_labels: Dict[str, int] = defaultdict(int)
    test_labels: Dict[str, int] = defaultdict(int)
    missing_annotations = 0

    for filename, subset in split_map.items():
        annot_path = audio_dir / f"{filename}.txt"
        if not annot_path.exists():
            missing_annotations += 1
            continue
        label = determine_label(annot_path)
        if subset == "train":
            train_labels[label] += 1
        else:
            test_labels[label] += 1

    if missing_annotations:
        print(f"[WARN] {missing_annotations} annotation files missing.", file=sys.stderr)

    all_classes = ["normal", "crackles", "wheezes", "both"]
    train_total = sum(train_labels.values())
    test_total = sum(test_labels.values())

    # Header
    print(f"{'Class':<12} {'Train':>8} {'Train%':>8} {'Test':>8} {'Test%':>8} {'Total':>8}")
    print("-" * 56)
    for cls in all_classes:
        t_tr = train_labels[cls]
        t_te = test_labels[cls]
        pct_tr = (t_tr / train_total * 100) if train_total else 0.0
        pct_te = (t_te / test_total * 100) if test_total else 0.0
        print(f"{cls:<12} {t_tr:>8} {pct_tr:>7.1f}% {t_te:>8} {pct_te:>7.1f}% {t_tr + t_te:>8}")
    print("-" * 56)
    print(f"{'TOTAL':<12} {train_total:>8}          {test_total:>8}          {train_total + test_total:>8}")

    # 5. Estimate approximate 60/40 split ratio on recordings
    ratio = n_train_files / total_files if total_files else 0
    print(f"\nApproximate train ratio (recordings): {ratio:.3f} "
          f"(expect ~0.60 for official 60/40 split)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_root = _resolve_data_root()
    if not data_root.exists():
        print(f"[ERROR] Data root does not exist: {data_root}", file=sys.stderr)
        print("Set ICBHI2017_ROOT env var or place the dataset at data/", file=sys.stderr)
        sys.exit(1)

    try:
        verify_split(data_root)
    except Exception as exc:
        print(f"\n[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)
