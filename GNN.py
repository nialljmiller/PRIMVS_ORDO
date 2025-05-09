import sys
import os
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

# === Parse command-line arguments ===
if len(sys.argv) < 2:
    print("Usage: python GNN.py <input_fits_file> [output_file]")
    print("Example: python GNN.py data.fits output_predictions.csv")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2] if len(sys.argv) > 2 else "fit_with_predictions.csv"

# === User Configuration ===
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
def load_fits_to_dataframe(fits_path):
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        # Try direct conversion without byteswap/newbyteorder
        try:
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Direct conversion failed: {e}")
            # Alternative approach using structured array
            try:
                return pd.DataFrame.from_records(data)
            except Exception as e:
                print(f"Structured array approach failed: {e}")
                # Last resort: manual field extraction
                columns = data.names
                df = pd.DataFrame()
                for col in columns:
                    df[col] = data[col]
                return df

# Use the new function for loading
try:
    train_df = load_fits_to_dataframe(TRAIN_PATH)
    fit_df = load_fits_to_dataframe(FIT_PATH)
    print(f"Successfully loaded {len(train_df)} training samples and {len(fit_df)} fitting samples")
except Exception as e:
    print(f"Failed to load FITS files: {e}")
    sys.exit(1)


    
# === Split into train and test sets ===
# You might need to adjust this logic based on your data structure
# For now, let's use 70% for training and 30% for testing
train_size = int(0.7 * len(data_df))
train_df = data_df[:train_size]
fit_df = data_df[train_size:]

print(f"Training set size: {len(train_df)}")
print(f"Testing set size: {len(fit_df)}")

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
if LABEL_COLUMN in train_df.columns:
    train_labels = train_df.loc[original_index[train_mask_array], LABEL_COLUMN].values
    label_encoder = LabelEncoder()
    encoded_labels = torch.tensor(label_encoder.fit_transform(train_labels), dtype=torch.long)

    y_all = torch.full((len(x_df_full),), -1, dtype=torch.long)
    y_all[torch.tensor(train_mask_array)] = encoded_labels
else:
    print(f"Warning: Label column '{LABEL_COLUMN}' not found. Running in inference-only mode.")
    y_all = torch.full((len(x_df_full),), -1, dtype=torch.long)
    label_encoder = None

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

# === Train Model if labels are available ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)
num_classes = len(label_encoder.classes_) if label_encoder is not None else 2
model = GCN(in_channels=x.shape[1], hidden_channels=64, out_channels=num_classes).to(device)

if label_encoder is not None:
    print(f"Training model with {num_classes} classes...")
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
else:
    print("No labels available for training. Skipping training phase.")

# === Inference ===
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    preds = logits.argmax(dim=1).cpu().numpy()
    probs = F.softmax(logits, dim=1).cpu().numpy()

# === Append Predictions ===
fit_mask = original_index >= len(train_df)
fit_df_cleaned = fit_df.iloc[original_index[fit_mask]].copy().reset_index(drop=True)

if label_encoder is not None:
    fit_df_cleaned["gnn_predicted_class"] = label_encoder.inverse_transform(preds[fit_mask])
else:
    fit_df_cleaned["gnn_predicted_class"] = preds[fit_mask]

fit_df_cleaned["gnn_confidence"] = probs[fit_mask].max(axis=1)

# === Save ===
fit_df_cleaned.to_csv(output_file, index=False)
print(f"Saved predictions to {output_file}")