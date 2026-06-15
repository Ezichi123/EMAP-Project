# EMAP Affective Computing Challenge — Arousal Prediction

Predicting moment-by-moment emotional **arousal** from EEG and physiological
signals using EEG scalp **topomaps** with a **CNN + GRU** sequence model.

**Best validation RMSE: 0.2993**

- **Author:** Ezichi Chimezie
- **Lab:** AI4PC Lab, Howard University
- **Advisor:** Dr. Saurav Aryal
- **Repository:** https://github.com/Ezichi123/EMAP-Project

---

## Approach in brief

Each 500 ms time-bin of 64-electrode, 4-band EEG is rendered into four scalp
**topographic maps** (Theta, Alpha, Beta, Gamma), stacked into a 4-channel
image. A CNN encodes each timestep's image; a bidirectional GRU models the
sequence of a video loop; a prediction head outputs one arousal value per
timestep. Physiological signals (heart rate, GSR), a loop indicator, and a
video-ID embedding are fused with the CNN features before the GRU.

The model is trained with a masked **MSE + first-difference** loss and tuned
with Optuna. Variable-length loops are padded and masked so that padded
timesteps never contribute to the loss or the reported metric.

---

## Repository structure

```
EMAP-Project/
├── 01_topomap_exploration.ipynb   # EDA + topomap generation pipeline
├── 02_dataset_and_model.ipynb     # dataset, CNN+GRU model, training, Optuna tuning
├── 03_raw_features.ipynb
├── fix_loops.ipynb                # one-time loop-label correction utility
├── model_fullseq.ipynb            # full-sequence (whole-trial) experiment
├── prediction.py                  # INFERENCE script — run trained model on test CSVs
├── make_features_csv.py           # generates the feature documentation CSV
├── selected_features.csv          # documented feature set used by the model
├── scalers/                       # saved StandardScalers (fit on train)
│   ├── scaler_phys.pkl
│   └── scaler_arousal.pkl
│   └── scaler_gsr.pkl
│   └── scaler_hr.pkl
│   └── participant_arousal_stats.csv
├── models/                        # trained checkpoints (gitignored; see note)
│   └── step3_optuna_tuning_best.pt   # best model (RMSE 0.2993)
│   └── optuna_step3_trials.csv
│   └── optuna_step4_trials.csv
│   └── results_log.jsonl
├── Model Architectures
│   ├── OptunaModelBest.py
│   └── v4_data_masking_baseline.py
│   └── v4_data_masking_loop_fixture.py
├── DATA.md                        # dataset structure notes
├── requirements.txt               # Python dependencies
└── README.md
```

> **Note on `models/` and data:** model checkpoints and the dataset are not
> committed to the repository (large binaries / private competition data).
> The best model checkpoint is included in the submission package alongside
> this code.

---

## Setup

Python 3.10 (developed in a conda environment). Install dependencies:

```bash
pip install -r requirements.txt
```

Key packages: `torch`, `mne`, `scikit-learn`, `joblib`, `pandas`, `numpy`,
`optuna`, `matplotlib`.


---

## Running the trained model (fast path)

To predict arousal on a folder of raw test CSVs using the provided checkpoint:

```bash
python prediction.py --test_dir path/to/test_csvs --out predictions.csv
```

`prediction.py` reproduces the full preprocessing pipeline end to end:
topomap generation → saved-scaler transform → CNN+GRU forward pass →
inverse-transform → per-row predictions. It requires:

- `models/step3_optuna_tuning_best.pt`
- `scalers/scaler_phys.pkl`, `scalers/scaler_arousal.pkl`

Output is a CSV with one predicted arousal value per input row
(`filename, participant_id, treatment_id, loop_num, row_idx, predicted_arousal`).

> Unknown video/treatment IDs are handled gracefully: if a test file uses a
> stimulus ID not seen in training, the model falls back to a neutral
> (mean) embedding rather than failing.

---

## Reproducing training from scratch (full path)

Requires the competition dataset (raw `train/` and `val/` CSVs).

1. **Generate topomaps** — run `01_topomap_exploration.ipynb`. This reads the
   raw CSVs and writes one `.npy` per file plus a `labels.csv`
   (per-row metadata and targets) into `topomap_data/`.
   *Note: this step renders an image per timestep and takes substantial time.*
2. **(Optional) Correct loop labels** — run `fix_loops.ipynb` once to align
   loop assignments to the original pre-trimming row counts.
3. **Train and tune** — run `02_dataset_and_model.ipynb`. This fits and saves
   the scalers, builds the dataset, trains the CNN+GRU, runs the Optuna search,
   and retrains the best configuration (the 0.2993 model).

---

## Results

Each change was tested as a clean, one-variable step.

| Experiment                    | Val RMSE |
|-------------------------------|:--------:|
| Baseline (per-loop CNN+GRU)   | 0.300    |
| + Categorical features        | 0.313    |
| + First-difference loss       | 0.309    |
| **+ Optuna tuning (best)**    | **0.2993** |
| Full-sequence input           | 0.311    |

**Best configuration:** per-loop CNN+GRU, hidden size 256, 1 bidirectional GRU
layer, dropout 0.59, masked MSE + first-difference loss (λ ≈ 0.88), Optuna-tuned.

### Evaluation note

All RMSE values use a **masked** metric: padded timesteps are excluded from
both the loss and the evaluation. Correcting an earlier metric that counted
padding (which had inflated the score to ~0.27) gave an honest baseline near
0.30, and every result is reported on this consistent metric.

### Key finding

Performance plateaus near 0.30 across all representations, features, losses,
and sequence structures, with a persistent train/validation gap. The dominant
factor is **generalization across participants** with widely differing arousal
baselines (validation participants are unseen by the model), pointing to
participant-adaptive modeling as the highest-value next step.

---

## Features

`selected_features.csv` documents the complete feature set the model uses.
This approach does **not** perform explicit feature selection: all 256 EEG
band-power features (64 electrodes × 4 bands) are used, encoded as topomap
image channels, together with heart rate and GSR, a loop indicator, and a
video-ID embedding. The two additional physiological signals present in the
raw data (IR plethysmograph, respiration) were not used by this model and are
marked accordingly.

---

## Submission contents

- `EMAP_Presentation.pptx` — approach and achieved accuracy
- `prediction.py` — inference script with full preprocessing
- `step3_optuna_tuning_best.pt` — best trained model (RMSE 0.2993)
- `selected_features.csv` — documented feature set
- Source code (this repository) for reproducibility