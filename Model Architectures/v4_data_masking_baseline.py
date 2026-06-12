# %% [markdown]
# # EMAP Dataset and Model Pipeline
# 
# This notebook builds the PyTorch Dataset class and model architecture
# for predicting arousal, heart rate, and GSR from EEG topomap images.
# ## Author
# Ezichi Chimezie
# ## Structure
# 1. Load and verify labels
# 2. Fit StandardScaler on train data
# 3. Build PyTorch Dataset class
# 4. Build CNN + Sequence model
# 5. Train and evaluate

# %% [markdown]
# ## Cell 1 — Imports and Label Loading
# Load all required libraries and read the labels CSV files generated
# by the topomap pipeline. These contain metadata (participant ID, 
# treatment ID, loop number) and targets (arousal, heart rate, GSR)
# for every row in the dataset.

# %%
import os, json, time

# ---- Change this ONE line for each new experiment ----
EXPERIMENT = "v5_videoID_embedding"
DESCRIPTION = "CNN+GRU, feature_dim=256, hidden_dim=128, dropout=0.3, 128x128 topomaps"

os.makedirs("models", exist_ok=True)
BEST_MODEL_PATH = f"models/{EXPERIMENT}_best.pt"
RESULTS_LOG = "models/results_log.jsonl"

# %%
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
import os

# ---- Paths ----
TOPOMAP_DIR = Path('topomap_data')
TRAIN_LABELS = TOPOMAP_DIR / 'train' / 'labels.csv'
VAL_LABELS = TOPOMAP_DIR / 'val' / 'labels.csv'

# ---- Load labels ----
train_labels = pd.read_csv(TRAIN_LABELS)
val_labels = pd.read_csv(VAL_LABELS)

print("Train samples:", len(train_labels))
print("Val samples:", len(val_labels))
print("\nColumns:", train_labels.columns.tolist())
print("\nSample:")
print(train_labels.head())

# %%
# Map treatment IDs to integer indices (T01 -> 0, T02 -> 1, ...)
all_treatments = sorted(train_labels['treatment_id'].unique())
treatment_to_idx = {t: i for i, t in enumerate(all_treatments)}
NUM_TREATMENTS = len(treatment_to_idx)
print(f"{NUM_TREATMENTS} treatments mapped")

# %% [markdown]
# ## Cell 2 — Fit StandardScaler on Train Data
# Fit StandardScaler on training set physiological features and labels.
# Scaler is fit on train only then applied to both train and val to prevent data leakage.
# Fitted scalers are saved to disk for later use during inference.

# %%
os.makedirs('scalers', exist_ok=True)

# ---- Physiological features to scale ----
phys_cols = ['heartrate', 'gsr']

# ---- Target columns to scale ----
target_cols = ['arousal', 'heartrate', 'gsr']

# ---- Fit scalers on train only ----
scaler_phys = StandardScaler()
scaler_arousal = StandardScaler()
scaler_hr = StandardScaler()
scaler_gsr = StandardScaler()

# Fit on train
scaler_phys.fit(train_labels[phys_cols])
scaler_arousal.fit(train_labels[['arousal']])
scaler_hr.fit(train_labels[['heartrate']])
scaler_gsr.fit(train_labels[['gsr']])

# Save scalers
joblib.dump(scaler_phys, 'scalers/scaler_phys.pkl')
joblib.dump(scaler_arousal, 'scalers/scaler_arousal.pkl')
joblib.dump(scaler_hr, 'scalers/scaler_hr.pkl')
joblib.dump(scaler_gsr, 'scalers/scaler_gsr.pkl')

print("Scalers fitted and saved")
print("\nPhysiological feature statistics (train):")
print(f"  Heart rate — mean: {scaler_phys.mean_[0]:.2f}, std: {scaler_phys.scale_[0]:.2f}")
print(f"  GSR        — mean: {scaler_phys.mean_[1]:.6f}, std: {scaler_phys.scale_[1]:.6f}")
print("\nTarget statistics (train):")
print(f"  Arousal    — mean: {scaler_arousal.mean_[0]:.4f}, std: {scaler_arousal.scale_[0]:.4f}")
print(f"  Heart rate — mean: {scaler_hr.mean_[0]:.2f}, std: {scaler_hr.scale_[0]:.2f}")
print(f"  GSR        — mean: {scaler_gsr.mean_[0]:.6f}, std: {scaler_gsr.scale_[0]:.6f}")



# %% [markdown]
# ## Cell 3 — PyTorch Dataset Class
# Builds a custom PyTorch Dataset that:
# - Loads topomap .npy files (shape: n_rows, 4, 128, 128)
# - Groups rows into loops (one sample = one loop)
# - Applies StandardScaler to physiological features and labels
# - Encodes loop number as a feature
# - Returns (topomap_sequence, phys_sequence, loop_num, target_sequence)

# %%
class EMAPDataset(Dataset):
    def __init__(self, labels_df, npy_dir, scaler_phys, scaler_target, 
                 target_col='arousal', split='train'):
        """
        Args:
            labels_df: DataFrame with metadata and labels
            npy_dir: Path to folder containing .npy files
            scaler_phys: fitted StandardScaler for physiological features
            scaler_target: fitted StandardScaler for target label
            target_col: one of 'arousal', 'heartrate', 'gsr'
            split: 'train' or 'val'
        """
        self.npy_dir = Path(npy_dir)
        self.scaler_phys = scaler_phys
        self.scaler_target = scaler_target
        self.target_col = target_col
        self.split = split
        
        # Group by file and loop number — one sample per loop per file
        self.samples = []
        
        grouped = labels_df.groupby(['filename', 'loop_num'])
        
        for (filename, loop_num), group in grouped:
            group = group.sort_values('bin_within_loop').reset_index(drop=True)

            treatment_id = group['treatment_id'].iloc[0]   # all rows in group share it
            
            self.samples.append({
                'filename': filename,
                'loop_num': loop_num,
                'treatment_idx': treatment_to_idx[treatment_id],
                'row_indices': group['row_idx'].values,
                'phys_features': group[['heartrate', 'gsr']].values,
                'targets': group[target_col].values
            })
        
        print(f"{split} dataset: {len(self.samples)} samples "
              f"({len(labels_df['filename'].unique())} files × ~4 loops)")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        npy_path = self.npy_dir / f"{sample['filename']}.npy"
        full_sequence = np.load(npy_path)

        row_indices = sample['row_indices']
        topomap_seq = full_sequence[row_indices]

        phys_scaled = self.scaler_phys.transform(
            pd.DataFrame(sample['phys_features'], columns=['heartrate', 'gsr'])
        )

        targets_scaled = self.scaler_target.transform(
            pd.DataFrame(sample['targets'].reshape(-1, 1), columns=[self.target_col])
        ).flatten()

        # ---- Record real length BEFORE padding ----
        real_len = topomap_seq.shape[0]
        max_len = 38

        if real_len < max_len:
            pad = max_len - real_len
            topomap_seq = np.pad(topomap_seq, ((0, pad), (0, 0), (0, 0), (0, 0)))
            phys_scaled = np.pad(phys_scaled, ((0, pad), (0, 0)))
            targets_scaled = np.pad(targets_scaled, (0, pad))
        elif real_len > max_len:
            topomap_seq = topomap_seq[:max_len]
            phys_scaled = phys_scaled[:max_len]
            targets_scaled = targets_scaled[:max_len]
            real_len = max_len

        # ---- Build mask: 1 for real timesteps, 0 for padding ----
        mask = np.zeros(max_len, dtype=np.float32)
        mask[:real_len] = 1.0

        loop_num_normalized = (sample['loop_num'] - 1) / 3.0

        topomap_tensor = torch.FloatTensor(topomap_seq)
        phys_tensor = torch.FloatTensor(phys_scaled)
        loop_tensor = torch.FloatTensor([loop_num_normalized])
        target_tensor = torch.FloatTensor(targets_scaled)
        mask_tensor = torch.FloatTensor(mask)
        treatment_tensor = torch.LongTensor([sample['treatment_idx']])  # (1,)

        return topomap_tensor, phys_tensor, loop_tensor, treatment_tensor, target_tensor, mask_tensor

        

# %%
# ---- Create train and val datasets ----
train_dataset = EMAPDataset(
    labels_df=train_labels,
    npy_dir=TOPOMAP_DIR / 'train' / 'npy',
    scaler_phys=scaler_phys,
    scaler_target=scaler_arousal,
    target_col='arousal',
    split='train'
)

val_dataset = EMAPDataset(
    labels_df=val_labels,
    npy_dir=TOPOMAP_DIR / 'val' / 'npy',
    scaler_phys=scaler_phys,
    scaler_target=scaler_arousal,
    target_col='arousal',
    split='val'
)

# %% [markdown]
# ## Cell 4 — DataLoaders
# Wraps the Dataset in PyTorch DataLoaders for batched training.
# DataLoader handles shuffling, batching, and parallel data loading.

# %%
# ---- Create DataLoaders ----
BATCH_SIZE = 32
NUM_WORKERS = 4

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True  # faster GPU transfer
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,   # no shuffling for val
    num_workers=NUM_WORKERS,
    pin_memory=True
)

print(f"Batch size: {BATCH_SIZE}")
print(f"Train batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")

# ---- Verify one batch ----
topomap, phys, loop_num, treatment,target, mask= next(iter(train_loader))
print(f"\nOne batch shapes:")
print(f"  Topomap: {topomap.shape}")    # (32, 38, 4, 128, 128)
print(f"  Phys: {phys.shape}")          # (32, 38, 2)
print(f"  Loop num: {loop_num.shape}")  # (32, 1)
print(f"  Treatment: {treatment.shape}") # (32, 1)
print(f"  Target: {target.shape}")      # (32, 38)
print(f"  Mask: {mask.shape}")         # (32, 38)
print(f"  Treatment values: {treatment[:5].flatten().tolist()}")

# %% [markdown]
# ## Cell 5 — Model Architecture
# Defines the CNN + GRU hybrid model:
# - CNN Encoder extracts spatial features from each topomap image
# - GRU Sequence Model captures temporal dependencies across time steps
# - Prediction Head outputs one arousal value per time step

# %%
import torch
import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, in_channels=4, feature_dim=256):
        """
        CNN that extracts spatial features from one topomap image.
        Input: (batch, 4, 128, 128)
        Output: (batch, feature_dim)
        """
        super(CNNEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 128 → 64
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 64 → 32
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 32 → 16
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 16 → 1x1
        )
        
        self.fc = nn.Linear(256, feature_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x: (batch, 4, 128, 128)
        x = self.encoder(x)           # (batch, 256, 1, 1)
        x = x.flatten(1)              # (batch, 256)
        x = self.dropout(x)
        x = self.fc(x)                # (batch, feature_dim)
        return x


class EEGArousalModel(nn.Module):
    def __init__(self, cnn_feature_dim=256, hidden_dim=128,
                 num_gru_layers=2, dropout=0.3, use_treatment=True):
        super().__init__()
        self.cnn = CNNEncoder(in_channels=4, feature_dim=cnn_feature_dim)
        self.use_treatment = use_treatment
        if use_treatment:
            self.treatment_emb = nn.Embedding(NUM_TREATMENTS, 8)
        extra = 8 if use_treatment else 0
        gru_input_dim = cnn_feature_dim + 2 + 1 + extra

        self.gru = nn.GRU(gru_input_dim, hidden_dim, num_gru_layers,
                          batch_first=True, dropout=dropout, bidirectional=True)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
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

# %% [markdown]
# ## Cell 6 — Model Instantiation and Verification
# Creates an instance of the model, moves it to GPU, and verifies
# it produces the correct output shape with a dummy batch.

# %%
# ---- Set device ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# ---- Instantiate model ----
model = EEGArousalModel(
    cnn_feature_dim=256,
    hidden_dim=128,
    num_gru_layers=2,
    dropout=0.3
).to(device)

# ---- Count parameters ----
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ---- Test with a dummy batch ----
dummy_topomap = torch.randn(4, 38, 4, 128, 128).to(device)
dummy_phys = torch.randn(4, 38, 2).to(device)
dummy_loop = torch.randn(4, 1).to(device)
dummy_treatment = torch.randint(0, NUM_TREATMENTS, (4, 1)).to(device)

with torch.no_grad():
    output = model(dummy_topomap, dummy_phys, dummy_loop, dummy_treatment)

print(f"\nDummy input shapes:")
print(f"  Topomap: {dummy_topomap.shape}")
print(f"  Phys: {dummy_phys.shape}")
print(f"  Loop num: {dummy_loop.shape}")
print(f"  Treatment: {dummy_treatment.shape}")
print(f"\nModel output shape: {output.shape}")  # should be (4, 38)
print(f"Output range: {output.min():.3f} to {output.max():.3f}")



# %% [markdown]
# ## Cell 7 — Training Loop
# Trains the model using MSE loss and Adam optimizer.
# Evaluates on validation set after each epoch.
# Saves the best model based on validation RMSE.

# %%
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

# ---- Training configuration ----
EPOCHS = 50
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = 'models/arousal_model.pt'
os.makedirs('models', exist_ok=True)

# ---- Loss function and optimizer ----
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for topomap, phys, loop_num, treatment, target, mask in loader:
        topomap = topomap.to(device)
        phys = phys.to(device)
        loop_num = loop_num.to(device)
        treatment = treatment.to(device)
        target = target.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        output = model(topomap, phys, loop_num, treatment)

        sq_err = ((output - target) ** 2) * mask
        loss = sq_err.sum() / mask.sum()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

def evaluate(model, loader, criterion, device, scaler_target):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for topomap, phys, loop_num, treatment, target, mask in loader:
            topomap = topomap.to(device)
            phys = phys.to(device)
            loop_num = loop_num.to(device)
            treatment = treatment.to(device)
            target = target.to(device)
            mask = mask.to(device)

            output = model(topomap, phys, loop_num, treatment)

            sq_err = ((output - target) ** 2) * mask
            loss = sq_err.sum() / mask.sum()
            total_loss += loss.item()

            mask_bool = mask.bool()
            all_preds.append(output[mask_bool].cpu().numpy())
            all_targets.append(target[mask_bool].cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    all_preds_orig = scaler_target.inverse_transform(all_preds.reshape(-1, 1)).flatten()
    all_targets_orig = scaler_target.inverse_transform(all_targets.reshape(-1, 1)).flatten()

    rmse = math.sqrt(np.mean((all_preds_orig - all_targets_orig) ** 2))
    return total_loss / len(loader), rmse

# %% [markdown]
# ## Cell 8 — Main Training Loop
# Runs training for the specified number of epochs.
# Tracks train loss, val loss, and val RMSE.
# Saves the best model based on lowest validation RMSE.

# %%
# ---- Training history ----
history = {
    'train_loss': [],
    'val_loss': [],
    'val_rmse': []
}

best_val_rmse = float('inf')
best_epoch = 0

print(f"Starting training for {EPOCHS} epochs...")
print(f"{'Epoch':>6} {'Train Loss':>12} {'Val Loss':>10} {'Val RMSE':>10} {'LR':>10}")
print("-" * 55)
best_val_rmse = float("inf")
best_epoch = 0
for epoch in range(1, EPOCHS + 1):
    # ---- Train ----
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    
    # ---- Evaluate ----
    val_loss, val_rmse = evaluate(model, val_loader, criterion, device, scaler_arousal)
    
    # ---- Update scheduler ----
    scheduler.step(val_loss)
    
    # ---- Save history ----
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_rmse'].append(val_rmse)
    
    # ---- Save best model ----
    if val_rmse < best_val_rmse:
        best_val_rmse = val_rmse
        best_epoch = epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_rmse': val_rmse,
        }, BEST_MODEL_PATH)
        print(f"  ↳ new best ({val_rmse:.6f}) saved to {BEST_MODEL_PATH}")
    # ---- Get current learning rate ----
    current_lr = optimizer.param_groups[0]['lr']
    
    # ---- Print progress ----
    print(f"{epoch:>6} {train_loss:>12.6f} {val_loss:>10.6f} {val_rmse:>10.6f} {current_lr:>10.2e}")

print("-" * 55)
print(f"\nTraining complete!")
print(f"Best Val RMSE: {best_val_rmse:.6f} at epoch {best_epoch}")

# %%
# After training completes, log the results
record = {
    "experiment": EXPERIMENT,
    "description": DESCRIPTION,
    "best_val_rmse": round(best_val_rmse, 6),
    "best_epoch": best_epoch,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "timestamp": time.strftime("%Y-%m-%d %H:%M"),
}
with open(RESULTS_LOG, "a") as f:
    f.write(json.dumps(record) + "\n")

print(f"Logged {EXPERIMENT}: best RMSE {best_val_rmse:.6f} (epoch {best_epoch})")

# %%
train_treatments = set(train_labels['treatment_id'].unique())
val_treatments = set(val_labels['treatment_id'].unique())

print(f"Train treatments: {len(train_treatments)} -> {sorted(train_treatments)}")
print(f"Val treatments:   {len(val_treatments)} -> {sorted(val_treatments)}")
print(f"In val but NOT train: {val_treatments - train_treatments}")
print(f"In train but NOT val: {train_treatments - val_treatments}")

# %%
model_0301 = EEGArousalModel(
    cnn_feature_dim=256,
    hidden_dim=128,
    num_gru_layers=2,
    dropout=0.3,
    use_treatment=False
).to(device)

ckpt = torch.load("models/v4_data_masking_best.pt", map_location=device)
model_0301.load_state_dict(ckpt["model_state_dict"])
model_0301.eval()
print("Loaded. Checkpoint val_rmse:", ckpt.get("val_rmse"))


