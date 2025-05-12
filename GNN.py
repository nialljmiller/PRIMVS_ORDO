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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from vis import *

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
    "true_period",
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
    data_df = load_fits_to_dataframe(input_file)
except Exception as e:
    print(f"Failed to load FITS files: {e}")
    sys.exit(1)


# Add this after loading the data
def get_max_prob_class(row):
    # Define the class probability columns
    class_cols = ['PQSO', 'PGal', 'Pstar', 'PWD', 'Pbin']
    # Get the column with max probability
    max_prob_col = max(class_cols, key=lambda col: row[col])
    # Return the class name (remove the 'P' prefix)
    return max_prob_col[1:]  # Remove 'P' from 'PQSO', 'PGal', etc.

# Apply this function to create a new 'true_class' column
data_df['true_class'] = data_df.apply(get_max_prob_class, axis=1)



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
    for epoch in range(2000):
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
# Identify indices corresponding to test set
fit_indices = [i for i, idx in enumerate(original_index) if idx >= len(train_df)]
fit_preds = preds[fit_indices]
fit_probs = probs[fit_indices]

# Create output dataframe with predictions
fit_df_cleaned = fit_df.copy()
if label_encoder is not None:
    fit_df_cleaned["gnn_predicted_class"] = label_encoder.inverse_transform(fit_preds)
else:
    fit_df_cleaned["gnn_predicted_class"] = fit_preds
fit_df_cleaned["gnn_confidence"] = fit_probs.max(axis=1)

# === Save ===
fit_df_cleaned.to_csv(output_file, index=False)
print(f"Saved predictions to {output_file}")











# Usage
# Assuming fit_df_cleaned has true and predicted labels
y_true = fit_df_cleaned[LABEL_COLUMN]  # True labels
y_pred = fit_df_cleaned["gnn_predicted_class"]  # Predicted labels
class_names = label_encoder.classes_  # Class names
plot_classification_performance(y_true, y_pred, class_names)











# Usage
confidences = fit_df_cleaned["gnn_confidence"].values
predictions = fit_df_cleaned["gnn_predicted_class"].map(
    {class_name: i for i, class_name in enumerate(class_names)}).values
plot_confidence_distribution(confidences, predictions, class_names)










# Usage example - add this to your main code
fit_indices = np.where(original_index >= len(train_df))[0]  # Get indices of test samples in the dataset
test_labels = fit_df_cleaned[LABEL_COLUMN].map(
    {class_name: i for i, class_name in enumerate(class_names)}).values
visualize_embeddings(model, data, test_labels, class_names, fit_indices)









# Usage
feature_importance_analysis(model, data, valid_features)







# Usage
node_labels = y_all.cpu().numpy()
node_labels = np.array([label if label != -1 else len(class_names) for label in node_labels])
class_names_with_unknown = list(class_names) + ["Unknown"]
visualize_knn_graph_sample(edge_index, node_labels, class_names_with_unknown)












# Select key astronomical features
astronomical_features = [
    "parallax", "pmra", "pmdec", "MAD", "true_amplitude", "true_period",
    "j_med_mag-ks_med_mag", "h_med_mag-ks_med_mag", "z_med_mag-ks_med_mag"
]

# Filter available features
available_astro_features = [f for f in astronomical_features if f in fit_df_cleaned.columns]
plot_feature_class_correlations(fit_df_cleaned, available_astro_features, LABEL_COLUMN)






# Check if required columns exist
plot_period_amplitude(fit_df_cleaned, LABEL_COLUMN)


