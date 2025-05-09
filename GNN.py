# GNN Classification Script for Stellar Sources using FITS files
# Requires: torch, torch_geometric, pandas, scikit-learn, astropy

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import kneighbors_graph
from astropy.io import fits

# === User Configuration ===
TRAIN_PATH = "train.fits"
FIT_PATH = "fit.fits"
FEATURE_COLUMNS = [str(i) for i in range(128)] + [
    "MAD", "eta", "true_amplitude", "mean_var", "std_nxs", "range_cum_sum", "max_slope",
    "percent_amp", "stet_k", "roms", "lag_auto", "Cody_M", "AD",
    "ls_period1", "ls_y_y_0", "ls_peak_width_0",
    "pdm_period1", "pdm_y_y_0",
    "ce_period1", "ce_y_y_0",
    "j_med_mag-ks_med_mag", "h_med_mag-ks_med_mag", "z_med_mag-ks_med_mag",
    "parallax", "pmra", "pmdec", "chisq", "uwe"
]  # Selected science features
LABEL_COLUMN = "true_class"
K_NEIGHBORS = 15

# === Load FITS Files ===
with fits.open(TRAIN_PATH) as hdul:
    train_data = hdul[1].data
    train_df = pd.DataFrame(train_data.byteswap().newbyteorder())

with fits.open(FIT_PATH) as hdul:
    fit_data = hdul[1].data
    fit_df = pd.DataFrame(fit_data.byteswap().newbyteorder())

# === Validate and Prepare Feature Matrix ===
common_features = [col for col in FEATURE_COLUMNS if col in train_df.columns and col in fit_df.columns]
x_df_full = pd.concat([train_df[common_features], fit_df[common_features]], ignore_index=True)

# === Handle NaNs or Infs and clip ===
x_df_full.replace([np.inf, -np.inf], np.nan, inplace=True)
clip_low = x_df_full.quantile(0.001)
clip_high = x_df_full.quantile(0.999)
x_df_full = x_df_full.clip(lower=clip_low, upper=clip_high, axis=1)
x_df_full.replace([np.inf, -np.inf], np.nan, inplace=True)

# === Drop rows with NaNs and reset index (preserve original index for alignment)
initial_len = len(x_df_full)
x_df_full = x_df_full.dropna().reset_index(drop=False)
print(f"Dropped {initial_len - len(x_df_full)} rows due to NaNs/infs. Remaining: {len(x_df_full)}")

# === Recompute features that survived the dropna ===
valid_features = [col for col in common_features if col in x_df_full.columns and not x_df_full[col].isna().any()]

# === Normalize Features ===
scaler = StandardScaler()
x = scaler.fit_transform(x_df_full[valid_features].values)
x = torch.tensor(x, dtype=torch.float)

# === Build Labels ===
original_index = x_df_full['index'].values
train_mask_array = original_index < len(train_df)
label_encoder = LabelEncoder()
train_labels = train_df.loc[original_index[train_mask_array], LABEL_COLUMN].values
encoded_labels = torch.tensor(label_encoder.fit_transform(train_labels), dtype=torch.long)

y_all = torch.full((len(x_df_full),), -1, dtype=torch.long)
y_all[torch.tensor(train_mask_array)] = encoded_labels

# === Create KNN Graph ===
knn = kneighbors_graph(x, n_neighbors=K_NEIGHBORS, include_self=True)
edge_index = torch.tensor(knn.nonzero(), dtype=torch.long)

# === GCN Model Definition ===
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

# === Prepare Data Object ===
data = Data(x=x, edge_index=edge_index, y=y_all)
train_mask = data.y != -1

# === Train Model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
model = GCN(in_channels=x.shape[1], hidden_channels=64, out_channels=len(label_encoder.classes_)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[train_mask], data.y[train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")

# === Inference ===
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    preds = logits.argmax(dim=1).cpu().numpy()
    probs = F.softmax(logits, dim=1).cpu().numpy()

# === Append Predictions ===
fit_mask = original_index >= len(train_df)
fit_df_cleaned = fit_df.iloc[original_index[fit_mask]].copy().reset_index(drop=True)
fit_df_cleaned["gnn_predicted_class"] = label_encoder.inverse_transform(preds[fit_mask])
fit_df_cleaned["gnn_confidence"] = probs[fit_mask].max(axis=1)

# === Save ===
fit_df_cleaned.to_csv("fit_with_predictions.csv", index=False)
print("Saved predictions to fit_with_predictions.csv")
