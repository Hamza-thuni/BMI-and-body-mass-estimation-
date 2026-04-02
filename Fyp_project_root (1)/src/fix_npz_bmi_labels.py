"""
fix_npz_bmi_labels.py
---------------------
Patches the BMI values in the already-extracted V7 .npz feature files.

The original _parse_bmi in extract_features_v7.py used filter(str.isdigit, ...)
which absorbed the '1' from ' (1)' copy-suffixes in dataset filenames, inflating
weight by 10x and producing unrealistic BMI values (400-450).

This script reloads each split .npz, recomputes BMI from the stored filenames
using the FIXED formula, and overwrites the file in-place.

Usage:
    cd src
    python fix_npz_bmi_labels.py
"""

import re
import sys
import os
import numpy as np

# Add src to path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from project_paths import features_dir


# ---------------------------------------------------------------------------
# Corrected BMI parser (identical fix applied to extract_features_v7.py)
# ---------------------------------------------------------------------------

def _parse_bmi_fixed(fname: str) -> float:
    """
    Parse BMI using re.search to grab only the FIRST digit run per field,
    preventing ' (1)' suffixes from being concatenated into the number.
    """
    name  = fname.split(".")[0]
    parts = name.split("_")
    m3 = re.search(r'\d+', parts[3])
    m4 = re.search(r'\d+', parts[4])
    if m3 is None or m4 is None:
        raise ValueError(f"digit group not found in {fname!r}")
    h_raw = int(m3.group())
    w_raw = int(m4.group())
    h_m  = h_raw / 100_000.0   # mm → m
    w_kg = w_raw / 100_000.0   # g*10 → kg
    bmi  = w_kg / (h_m ** 2 + 1e-8)
    if not (10.0 < bmi < 80.0):
        raise ValueError(
            f"Implausible BMI {bmi:.1f} for {fname!r} "
            f"(h={h_m:.3f} m, w={w_kg:.1f} kg)"
        )
    return bmi


# ---------------------------------------------------------------------------
# Patch each split
# ---------------------------------------------------------------------------

def patch_split(npz_path: str) -> None:
    print(f"\nPatching: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)

    names    = data["names"]      # array of filename strings
    old_bmi  = data["bmi"]

    print(f"  Samples : {len(names)}")
    print(f"  Old BMI : min={old_bmi.min():.2f}  max={old_bmi.max():.2f}  "
          f"mean={old_bmi.mean():.2f}")

    new_bmi  = np.empty_like(old_bmi)
    n_fixed  = 0
    n_skip   = 0

    for i, fname in enumerate(names):
        try:
            new_bmi[i] = _parse_bmi_fixed(str(fname))
            n_fixed += 1
        except Exception as e:
            # Keep old value so array stays aligned; will be filtered at training
            new_bmi[i] = old_bmi[i]
            n_skip += 1
            if n_skip <= 5:
                print(f"  [WARN] Could not reparse {fname!r}: {e}")

    print(f"  New BMI : min={new_bmi.min():.2f}  max={new_bmi.max():.2f}  "
          f"mean={new_bmi.mean():.2f}")
    print(f"  Fixed {n_fixed} / {len(names)}  |  Skipped {n_skip}")

    # Collect all arrays from the original file
    arrays = {k: data[k] for k in data.files}
    arrays["bmi"] = new_bmi                 # overwrite with corrected labels

    np.savez_compressed(npz_path, **arrays)
    print(f"  [OK] Saved (in-place): {npz_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    feat_dir = str(features_dir("v7"))
    print("=" * 60)
    print("V7 .npz BMI Label Patcher")
    print("=" * 60)

    splits = ["train", "val", "test"]
    for split in splits:
        path = os.path.join(feat_dir, f"{split}_v7.npz")
        if os.path.exists(path):
            patch_split(path)
        else:
            print(f"\n[SKIP] Not found: {path}")

    print("\n[OK] All patches applied. Now re-run train_v7_hybrid.py.")
