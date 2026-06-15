"""
prediction.py — EMAP Arousal Prediction (inference)

Runs the trained CNN+GRU model on raw test CSV files and writes predicted
arousal values. Reproduces the exact preprocessing used in training:
topomap generation, per-loop sequencing, saved scalers, masking.

Usage:
    python prediction.py --test_dir path/to/test_csvs --out predictions.csv

Artifacts required (same folder layout as the training repo):
    models/step3_optuna_tuning_best.pt
    scalers/scaler_phys.pkl
    scalers/scaler_arousal.pkl

Author: Ezichi Chimezie
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; no display needed
import matplotlib.pyplot as plt
import mne
import joblib
import torch
import torch.nn as nn

# ============================================================
# CONFIG — paths and constants (match training)
# ============================================================
MODEL_PATH = "models/step3_optuna_tuning_best.pt"
SCALER_PHYS_PATH = "scalers/scaler_phys.pkl"
SCALER_AROUSAL_PATH = "scalers/scaler_arousal.pkl"

IMG_SIZE = 128
MAX_LEN = 38
BANDS = ["Theta", "Alpha", "Beta", "Gamma"]

# Best (Optuna-tuned) architecture — the 0.2993 model
HIDDEN_DIM = 256
NUM_GRU_LAYERS = 1
DROPOUT = 0.59
USE_TREATMENT = True

# Treatment-ID map, reproduced exactly as training built it:
#   sorted(train treatment_ids) -> 0..23, for T01..T24
# Hardcoded so the script is self-contained (no train labels needed).
TREATMENTS = [f"T{i:02d}" for i in range(1, 25)]      # T01..T24
TREATMENT_TO_IDX = {t: i for i, t in enumerate(TREATMENTS)}
NUM_TREATMENTS = len(TREATMENT_TO_IDX)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# TOPOMAP GENERATION  (verbatim from training pipeline)
# ============================================================
def get_montage_and_electrodes(sample_df):
    """Build the montage 2D positions and electrode list from CSV headers,
    exactly as in the training generation code."""
    montage = mne.channels.make_standard_montage("standard_1005")
    montage_pos = montage.get_positions()["ch_pos"]

    eeg_cols = [c for c in sample_df.columns if c.startswith("EEG_")]
    electrodes = list(dict.fromkeys(
        [c.replace("EEG_", "").rsplit("_", 1)[0] for c in eeg_cols]
    ))
    pos_2d = np.array([[montage_pos[e][0], montage_pos[e][1]] for e in electrodes])
    return electrodes, pos_2d


def generate_topomap_image(row_data, electrodes, pos_2d, bands, img_size=128):
    """Generate a 4-channel topomap image from one row of EEG data.
    Returns (4, img_size, img_size). Identical to training."""
    channels = []
    for band in bands:
        band_values = np.array([row_data[f"EEG_{elec}_{band}"] for elec in electrodes])
        fig, ax = plt.subplots(figsize=(1, 1), dpi=img_size)
        ax.set_position([0, 0, 1, 1])
        mne.viz.plot_topomap(
            band_values, pos_2d, axes=ax, show=False,
            contours=0, cmap="RdBu_r", sensors=False, outlines="head",
        )
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        img = np.asarray(buf)[:, :, :3]
        img_gray = np.mean(img, axis=2) / 255.0
        channels.append(img_gray)
        plt.close(fig)
    return np.stack(channels, axis=0)


# ============================================================
# MODEL  (verbatim from training)
# ============================================================
class CNNEncoder(nn.Module):
    def __init__(self, in_channels=4, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(256, feature_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.encoder(x).flatten(1)
        return self.fc(self.dropout(x))


class EEGArousalModel(nn.Module):
    def __init__(self, cnn_feature_dim=256, hidden_dim=128,
                 num_gru_layers=2, dropout=0.3, use_treatment=True):
        super().__init__()
        self.cnn = CNNEncoder(in_channels=4, feature_dim=cnn_feature_dim)
        self.use_treatment = use_treatment
        if use_treatment:
            self.treatment_emb = nn.Embedding(NUM_TREATMENTS, 8)
        extra = 8 if use_treatment else 0
        gru_input_dim = cnn_feature_dim + 2 + 4 + extra
        self.gru = nn.GRU(gru_input_dim, hidden_dim, num_gru_layers,
                          batch_first=True,
                          dropout=dropout if num_gru_layers > 1 else 0.0,
                          bidirectional=True)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, topomap, phys, loop_num, treatment=None):
        b, seq_len, C, H, W = topomap.shape
        cnn_features = self.cnn(topomap.view(b * seq_len, C, H, W)).view(b, seq_len, -1)
        loop_expanded = loop_num.unsqueeze(1).expand(-1, seq_len, -1)
        feats = [cnn_features, phys, loop_expanded]
        if self.use_treatment:
            treat = self.treatment_emb(treatment.squeeze(-1)).unsqueeze(1).expand(-1, seq_len, -1)
            feats.append(treat)
        combined = torch.cat(feats, dim=-1)
        gru_out, _ = self.gru(combined)
        return self.prediction_head(gru_out).squeeze(-1)


# ============================================================
# PREPROCESS ONE FILE -> per-loop samples ready for the model
# ============================================================
def preprocess_file(csv_file, electrodes, pos_2d, scaler_phys, avg_treatment_idx):
    """Turn one raw test CSV into a list of per-loop inference samples.

    Mirrors training:
      - dropna (removes leading-NaN rows that lack a usable row)
      - trim end remainder so rows split into 4 equal loops
      - per-row topomaps; phys columns renamed heartrate_mean/GSR_mean -> heartrate/gsr
      - per-loop grouping, loop one-hot, pad/truncate to MAX_LEN, mask
    Returns: list of dicts with tensors + bookkeeping to map predictions back.
    """
    filename = csv_file.stem
    parts = filename.replace("Features_", "").split("-")
    participant_id = parts[0]
    treatment_id = parts[1] if len(parts) > 1 else "UNKNOWN"

    # Path-B fallback: unknown treatment ID -> average embedding index sentinel.
    if treatment_id in TREATMENT_TO_IDX:
        treatment_idx = TREATMENT_TO_IDX[treatment_id]
    else:
        treatment_idx = avg_treatment_idx  # handled specially downstream
        print(f"  [warn] unknown treatment '{treatment_id}' in {filename}; "
              f"using fallback embedding")

    df = pd.read_csv(csv_file)
    df_clean = df.dropna().reset_index(drop=True)
    total_rows = len(df_clean)
    if total_rows < 4:
        print(f"  [warn] {filename}: too few rows ({total_rows}); skipped")
        return []

    bins_per_loop = total_rows // 4
    remainder = total_rows % 4
    if remainder != 0:
        df_clean = df_clean.iloc[:total_rows - remainder].reset_index(drop=True)
        total_rows = len(df_clean)

    # per-row topomap + phys + loop assignment (generation-time logic)
    images, phys_rows, loop_nums, orig_row_idx = [], [], [], []
    for row_idx in range(total_rows):
        row = df_clean.iloc[row_idx]
        images.append(generate_topomap_image(row, electrodes, pos_2d, BANDS, IMG_SIZE))
        phys_rows.append([row["heartrate_mean"], row["GSR_mean"]])  # rename to heartrate/gsr
        loop_nums.append(min(row_idx // bins_per_loop, 3) + 1)
        orig_row_idx.append(row_idx)

    images = np.stack(images, axis=0)               # (total_rows, 4, 128, 128)
    phys_rows = np.array(phys_rows, dtype=np.float32)
    loop_nums = np.array(loop_nums)

    # scale phys with the SAVED scaler (transform only)
    phys_scaled_all = scaler_phys.transform(
        pd.DataFrame(phys_rows, columns=["heartrate", "gsr"])
    ).astype(np.float32)

    # group into loops -> one sample per loop
    samples = []
    for loop in [1, 2, 3, 4]:
        idx = np.where(loop_nums == loop)[0]
        if len(idx) == 0:
            continue
        seq = images[idx]
        phys = phys_scaled_all[idx]
        real_len = seq.shape[0]

        if real_len < MAX_LEN:
            pad = MAX_LEN - real_len
            seq = np.pad(seq, ((0, pad), (0, 0), (0, 0), (0, 0)))
            phys = np.pad(phys, ((0, pad), (0, 0)))
        elif real_len > MAX_LEN:
            seq, phys = seq[:MAX_LEN], phys[:MAX_LEN]
            idx = idx[:MAX_LEN]
            real_len = MAX_LEN

        mask = np.zeros(MAX_LEN, dtype=np.float32)
        mask[:real_len] = 1.0

        loop_oh = np.zeros(4, dtype=np.float32)
        loop_oh[loop - 1] = 1.0

        samples.append({
            "filename": filename,
            "participant_id": participant_id,
            "treatment_id": treatment_id,
            "loop_num": loop,
            "row_indices": [orig_row_idx[i] for i in idx[:real_len]],
            "topomap": torch.FloatTensor(seq),
            "phys": torch.FloatTensor(phys),
            "loop_oh": torch.FloatTensor(loop_oh),
            "treatment_idx": treatment_idx,
            "mask": mask,
            "real_len": real_len,
        })
    return samples


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", required=True, help="folder of raw test CSV files")
    ap.add_argument("--out", default="predictions.csv", help="output CSV path")
    args = ap.parse_args()

    test_dir = Path(args.test_dir)
    csv_files = sorted(test_dir.glob("*.csv"))
    if not csv_files:
        raise SystemExit(f"No CSV files found in {test_dir}")
    print(f"Found {len(csv_files)} test files")

    # ---- load saved scalers (transform only; never fit) ----
    scaler_phys = joblib.load(SCALER_PHYS_PATH)
    scaler_arousal = joblib.load(SCALER_AROUSAL_PATH)

    # ---- reconstruct model + load weights ----
    model = EEGArousalModel(
        cnn_feature_dim=256, hidden_dim=HIDDEN_DIM,
        num_gru_layers=NUM_GRU_LAYERS, dropout=DROPOUT,
        use_treatment=USE_TREATMENT,
    ).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model (checkpoint val_rmse={ckpt.get('val_rmse')})")

    # Path-B fallback: average embedding index — use 0 as sentinel; we override
    # the embedding with the mean vector at inference for unknown IDs.
    avg_treatment_idx = 0
    with torch.no_grad():
        mean_treat_vec = model.treatment_emb.weight.mean(dim=0, keepdim=True) \
            if USE_TREATMENT else None

    # electrodes/montage from the first file's headers
    sample_df = pd.read_csv(csv_files[0])
    electrodes, pos_2d = get_montage_and_electrodes(sample_df)

    results = []
    for fi, csv_file in enumerate(csv_files, 1):
        samples = preprocess_file(csv_file, electrodes, pos_2d, scaler_phys, avg_treatment_idx)
        for s in samples:
            topo = s["topomap"].unsqueeze(0).to(device)   # (1, MAX_LEN, 4, 128, 128)
            phys = s["phys"].unsqueeze(0).to(device)
            loop_oh = s["loop_oh"].unsqueeze(0).to(device)
            treat = torch.LongTensor([[s["treatment_idx"]]]).to(device)

            with torch.no_grad():
                # unknown treatment -> swap in mean embedding vector
                if USE_TREATMENT and s["treatment_id"] not in TREATMENT_TO_IDX:
                    orig = model.treatment_emb.weight.data[0].clone()
                    model.treatment_emb.weight.data[0] = mean_treat_vec.squeeze(0)
                    out = model(topo, phys, loop_oh, treat)
                    model.treatment_emb.weight.data[0] = orig
                else:
                    out = model(topo, phys, loop_oh, treat)

            pred_scaled = out.squeeze(0).cpu().numpy()[: s["real_len"]]
            pred = scaler_arousal.inverse_transform(
                pred_scaled.reshape(-1, 1)
            ).flatten()

            for ri, p in zip(s["row_indices"], pred):
                results.append({
                    "filename": s["filename"],
                    "participant_id": s["participant_id"],
                    "treatment_id": s["treatment_id"],
                    "loop_num": s["loop_num"],
                    "row_idx": ri,
                    "predicted_arousal": float(p),
                })
        if fi % 25 == 0 or fi == len(csv_files):
            print(f"  processed {fi}/{len(csv_files)} files")

    out_df = pd.DataFrame(results).sort_values(["filename", "row_idx"]).reset_index(drop=True)
    out_df.to_csv(args.out, index=False)
    print(f"\nWrote {len(out_df)} predictions to {args.out}")


if __name__ == "__main__":
    main()