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
if len(sys.argv) < 3:
    print("Usage: python GNN.py <training_fits_file> <testing_fits_file> [output_file]")
    print("Example: python GNN.py primvs.fits primvs_gaia.fits output_predictions.csv")
    sys.exit(1)

training_file = sys.argv[1]
testing_file = sys.argv[2]
output_file = sys.argv[3] if len(sys.argv) > 3 else "fit_with_predictions.csv"

# === User Configuration ===
# Base astronomical features
BASE_FEATURES = [
    "MAD", "eta", "true_amplitude", "mean_var", "std_nxs", "range_cum_sum", "max_slope",
    "percent_amp", "stet_k", "roms", "lag_auto", "Cody_M", "AD", "true_period",
    "j_med_mag-ks_med_mag", "h_med_mag-ks_med_mag", "z_med_mag-ks_med_mag",
    "parallax", "pmra", "pmdec", "chisq", "uwe"
]

# Embedding features (numbered dimensions from contrastive learning)
EMBEDDING_FEATURES = [str(i) for i in range(128)]

# All features to use
FEATURE_COLUMNS = EMBEDDING_FEATURES + BASE_FEATURES

# Classification labels
LABEL_COLUMN = "best_class_name"

# Graph and model configuration
K_NEIGHBORS = 8  # Reduced from 15 to save memory
MAX_CLASSES = 15  # Limit to top N classes to prevent memory issues
BATCH_SIZE = 4096  # For memory-efficient training
USE_CPU = False  # Set to True to force CPU usage

# === Load FITS Files ===
def load_fits_to_dataframe(fits_path):
    print(f"Loading data from {fits_path}...")
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        try:
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Direct conversion failed: {e}")
            try:
                return pd.DataFrame.from_records(data)
            except Exception as e:
                print(f"Structured array approach failed: {e}")
                columns = data.names
                df = pd.DataFrame()
                for col in columns:
                    df[col] = data[col]
                return df

# Load training data
try:
    train_df = load_fits_to_dataframe(training_file)
    print(f"Loaded training data: {len(train_df)} rows")
except Exception as e:
    print(f"Failed to load training file: {e}")
    sys.exit(1)

# Load testing data
try:
    test_df = load_fits_to_dataframe(testing_file)
    print(f"Loaded testing data: {len(test_df)} rows")
except Exception as e:
    print(f"Failed to load testing file: {e}")
    sys.exit(1)

# === Analyze classes ===
print("\nExamining true_class distribution in training data:")
train_classes = train_df[LABEL_COLUMN].value_counts()
print(f"Found {len(train_classes)} unique classes")
print(train_classes.head(10))

# Limit to most common classes to prevent memory issues
if len(train_classes) > MAX_CLASSES:
    common_classes = train_classes.nlargest(MAX_CLASSES).index.tolist()
    print(f"\nLimiting to {MAX_CLASSES} most common classes: {common_classes}")
    
    # Create new column with common classes or "Other"
    train_df['class_grouped'] = train_df[LABEL_COLUMN].apply(
        lambda x: x if x in common_classes else "Other"
    )
    test_df['class_grouped'] = test_df[LABEL_COLUMN].apply(
        lambda x: x if x in common_classes else "Other"
    )
    LABEL_COLUMN = 'class_grouped'
    
    # Show the distribution of the new classes
    print("\nDistribution after grouping:")
    print(train_df[LABEL_COLUMN].value_counts())

# === Identify common features across both datasets ===
train_features = set(train_df.columns)
test_features = set(test_df.columns)
common_features = list(train_features.intersection(test_features))
available_features = [f for f in FEATURE_COLUMNS if f in common_features]

print(f"\nFound {len(available_features)} common features across both datasets")
if len(available_features) < 10:
    print("WARNING: Very few features available. Identifying alternative features...")
    # Look for embedding features with different naming patterns
    potential_embeddings = [col for col in common_features if (
        col.startswith('z') or col.startswith('h') or col.startswith('u')
    ) and (
        'pca' in col.lower() or 'tsne' in col.lower() or 'umap' in col.lower()
    )]
    
    if potential_embeddings:
        print(f"Found {len(potential_embeddings)} potential embedding features")
        available_features += [col for col in potential_embeddings if col not in available_features]

print(f"Using {len(available_features)} features for model training")

# === Prepare Feature Matrices ===
# Process training data
X_train = train_df[available_features].copy()
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"NaN values in training data: {X_train.isna().sum().sum()}")

# Process testing data
X_test = test_df[available_features].copy()
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
print(f"NaN values in testing data: {X_test.isna().sum().sum()}")

# Compute statistics from training data only
train_min = X_train.quantile(0.001)
train_max = X_train.quantile(0.999)
train_mean = X_train.mean()

# Apply clipping and imputation to both datasets using training statistics
X_train = X_train.clip(lower=train_min, upper=train_max, axis=1)
X_train.fillna(train_mean, inplace=True)

X_test = X_test.clip(lower=train_min, upper=train_max, axis=1)
X_test.fillna(train_mean, inplace=True)

# Combine for standardization
X_combined = pd.concat([X_train, X_test], ignore_index=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_combined)

# Split back to train and test
X_train_scaled = X_scaled[:len(X_train)]
X_test_scaled = X_scaled[len(X_train):]

# Convert to tensors
x_train = torch.tensor(X_train_scaled, dtype=torch.float)
x_test = torch.tensor(X_test_scaled, dtype=torch.float)

# === Prepare Labels ===
label_encoder = LabelEncoder()
y_train = torch.tensor(label_encoder.fit_transform(train_df[LABEL_COLUMN]), dtype=torch.long)
num_classes = len(label_encoder.classes_)

print(f"\nUsing {num_classes} classes for training")
print("Class mapping:")
for i, cls in enumerate(label_encoder.classes_):
    print(f"  {i}: {cls}")

# === Create KNN Graph for Training Data ===
print("\nBuilding KNN graph...")
knn_train = kneighbors_graph(x_train, n_neighbors=K_NEIGHBORS, include_self=True)
edge_index_train = torch.tensor(knn_train.nonzero(), dtype=torch.long)
print(f"Created graph with {edge_index_train.shape[1]} edges")

# === GCN Model Definition ===
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        # Extract embeddings from first layer
        return self.conv1(x, edge_index).detach()

# === Setup Device ===
if USE_CPU:
    device = torch.device('cpu')
    print("Using CPU for computation (forced)")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

# === Prepare Training Data ===
data = Data(x=x_train, edge_index=edge_index_train, y=y_train)
data = data.to(device)

# === Initialize Model ===
hidden_dim = min(64, max(32, num_classes * 2))  # Adaptive hidden dimension
model = GCN(in_channels=x_train.shape[1], hidden_channels=hidden_dim, out_channels=num_classes).to(device)

# === Train Model ===
print(f"\nTraining model with {num_classes} classes...")
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Compute class weights for imbalanced classes
class_counts = np.bincount(y_train.cpu().numpy())
class_weights = 1.0 / (class_counts + 1e-10)  # Add small epsilon to avoid division by zero
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Batched training
model.train()
for epoch in range(150):
    total_loss = 0
    # Shuffle indices for better training
    train_indices = torch.randperm(len(y_train))
    num_batches = (len(train_indices) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in range(num_batches):
        batch_indices = train_indices[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[batch_indices], data.y[batch_indices], weight=class_weights)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Avg Loss: {total_loss/num_batches:.4f}")

# === Create KNN Graph for Testing Data ===
print("\nBuilding KNN graph for test data...")
# For efficiency, connect test nodes to their nearest training neighbors
combined_data = torch.cat([x_train, x_test], dim=0)
knn_combined = kneighbors_graph(combined_data.cpu().numpy(), n_neighbors=K_NEIGHBORS, include_self=True)
edge_index_combined = torch.tensor(knn_combined.nonzero(), dtype=torch.long)
print(f"Created combined graph with {edge_index_combined.shape[1]} edges")

# === Run Inference ===
model.eval()
print("\nRunning inference on test data...")
with torch.no_grad():
    test_data = Data(x=combined_data.to(device), edge_index=edge_index_combined.to(device))
    logits = model(test_data.x, test_data.edge_index)
    probs = F.softmax(logits, dim=1)
    
    # Extract predictions for test data
    test_indices = torch.arange(len(x_train), len(combined_data))
    test_logits = logits[test_indices]
    test_probs = probs[test_indices]
    
    preds = test_logits.argmax(dim=1).cpu().numpy()
    confs = test_probs.max(dim=1)[0].cpu().numpy()

# === Add Predictions to Test DataFrame ===
test_df_result = test_df.copy()
test_df_result["gnn_predicted_class_id"] = preds
test_df_result["gnn_predicted_class"] = label_encoder.inverse_transform(preds)
test_df_result["gnn_confidence"] = confs

# === Save Results ===
test_df_result.to_csv(output_file, index=False)
print(f"\nSaved predictions to {output_file}")

# === Evaluate if we have ground truth ===
if LABEL_COLUMN in test_df_result.columns:
    print("\nEvaluation on test set:")
    y_true = test_df_result[LABEL_COLUMN]
    y_pred = test_df_result["gnn_predicted_class"]
    
    # Print classification metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_true, y_pred))
    
    # Create visualizations
    class_names = label_encoder.classes_
    try:
        # Create plots if vis.py functions are available
        plot_classification_performance(y_true, y_pred, class_names)
        print("Saved confusion matrix visualization to confusion_matrix.png")
        
        # Plot confidence distribution
        plot_confidence_distribution(confs, preds, class_names)
        print("Saved confidence distribution to confidence_distribution.png")
        
        # Plot period-amplitude diagram if those features exist
        if "true_period" in test_df_result.columns and "true_amplitude" in test_df_result.columns:
            plot_period_amplitude(test_df_result, "gnn_predicted_class")
            print("Saved period-amplitude visualization to period_amplitude.png")
        
        # Feature importance analysis
        feature_importance_analysis(model, data, available_features)
        print("Saved feature importance visualization to feature_importance.png")
    except Exception as e:
        print(f"Visualization error: {e}")
        
print("\nProcessing complete!")