"""
make_features_csv.py — generate the descriptive feature list (Framing A)

Reads the actual column names from a sample data CSV and writes
selected_features.csv documenting every feature the model uses.
The approach uses the FULL feature set (no selection): EEG band powers
are encoded as topomap images, physiological signals are fed directly,
and loop/treatment are engineered indicators.

Usage:
    python make_features_csv.py --sample train/Features_P001-T01.csv --out selected_features.csv
"""

import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True, help="any one data CSV to read column names from")
    ap.add_argument("--out", default="selected_features.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.sample, nrows=1)  # only need the header
    cols = df.columns.tolist()

    rows = []

    # ---- 1. EEG band-power features (used as topomap image channels) ----
    eeg_cols = [c for c in cols if c.startswith("EEG_")]
    for c in eeg_cols:
        # c looks like EEG_<electrode>_<band>
        body = c.replace("EEG_", "")
        electrode, band = body.rsplit("_", 1)
        rows.append({
            "feature_name": c,
            "feature_type": "EEG_band_power",
            "electrode": electrode,
            "band": band,
            "representation": "topomap image channel",
            "used": "yes",
        })

    # ---- 2. Physiological features (fed directly, standardized) ----
    # The model uses heart rate and GSR. (IRPleth / Respiration are present
    # in the raw data but were NOT used by this model.)
    phys_used = {
        "heartrate_mean": "heart rate (BPM)",
        "GSR_mean": "skin conductance / EDA",
    }
    for c in cols:
        if c in phys_used:
            rows.append({
                "feature_name": c,
                "feature_type": "physiological",
                "electrode": "",
                "band": "",
                "representation": "standardized scalar (per timestep)",
                "used": "yes",
            })

    # physiological signals present but not used by this model
    phys_unused = ["IRPleth_mean", "Respir_mean"]
    for c in cols:
        if c in phys_unused:
            rows.append({
                "feature_name": c,
                "feature_type": "physiological",
                "electrode": "",
                "band": "",
                "representation": "not used by this model",
                "used": "no",
            })

    # ---- 3. Engineered features ----
    rows.append({
        "feature_name": "loop_num",
        "feature_type": "engineered",
        "electrode": "",
        "band": "",
        "representation": "one-hot (4 video repetitions)",
        "used": "yes",
    })
    rows.append({
        "feature_name": "treatment_id",
        "feature_type": "engineered",
        "electrode": "",
        "band": "",
        "representation": "learned embedding (24 video stimuli)",
        "used": "yes",
    })

    out = pd.DataFrame(rows, columns=[
        "feature_name", "feature_type", "electrode", "band", "representation", "used"
    ])
    out.to_csv(args.out, index=False)

    # summary
    n_eeg = (out["feature_type"] == "EEG_band_power").sum()
    n_phys_used = ((out["feature_type"] == "physiological") & (out["used"] == "yes")).sum()
    n_eng = (out["feature_type"] == "engineered").sum()
    print(f"Wrote {len(out)} rows to {args.out}")
    print(f"  EEG band-power features: {n_eeg} (64 electrodes x 4 bands)")
    print(f"  Physiological used: {n_phys_used}  |  engineered: {n_eng}")
    print(f"  Note: approach uses the FULL EEG set (no feature selection); "
          f"features are encoded as topomaps rather than pruned.")


if __name__ == "__main__":
    main()