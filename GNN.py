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
    for epoch in range(20):
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








import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

def plot_classification_performance(y_true, y_pred, class_names):
    # Generate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    #plt.show()
    
    # Print classification report
    print(classification_report(y_true, y_pred, target_names=class_names))

# Usage
# Assuming fit_df_cleaned has true and predicted labels
y_true = fit_df_cleaned[LABEL_COLUMN]  # True labels
y_pred = fit_df_cleaned["gnn_predicted_class"]  # Predicted labels
class_names = label_encoder.classes_  # Class names
plot_classification_performance(y_true, y_pred, class_names)








def plot_confidence_distribution(confidences, predictions, class_names):
    plt.figure(figsize=(12, 6))
    for i, class_name in enumerate(class_names):
        class_conf = confidences[predictions == i]
        if len(class_conf) > 0:
            sns.kdeplot(class_conf, label=f"{class_name} (n={len(class_conf)})")
    
    plt.xlabel('Model Confidence')
    plt.ylabel('Density')
    plt.title('Distribution of Model Confidence by Predicted Class')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('confidence_distribution.png', dpi=300)
    #plt.show()

# Usage
confidences = fit_df_cleaned["gnn_confidence"].values
predictions = fit_df_cleaned["gnn_predicted_class"].map(
    {class_name: i for i, class_name in enumerate(class_names)}).values
plot_confidence_distribution(confidences, predictions, class_names)






def visualize_embeddings(model, data, labels, class_names, test_indices):
    """
    Visualize the embeddings from the GNN model using t-SNE and PCA projections.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The trained GNN model
    data : torch_geometric.data.Data
        The full graph data object
    labels : numpy.ndarray
        The class labels for the test set
    class_names : list
        List of class names
    test_indices : numpy.ndarray
        Indices of test set samples in the full dataset
        
    Returns:
    --------
    None, saves visualization files
    """
    # Extract embeddings from penultimate layer
    model.eval()
    with torch.no_grad():
        # Forward pass through first GCN layer only
        embeddings = model.conv1(data.x, data.edge_index)
        embeddings = F.relu(embeddings)
        embeddings = embeddings.cpu().numpy()
    
    # Filter embeddings to include only test set
    test_embeddings = embeddings[test_indices]
    
    # Apply dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced_embeddings = tsne.fit_transform(test_embeddings)
    
    # Plot
    plt.figure(figsize=(12, 10))
    for i, class_name in enumerate(class_names):
        mask = labels == i
        if np.any(mask):  # Only plot if we have at least one point
            plt.scatter(reduced_embeddings[mask, 0], reduced_embeddings[mask, 1], 
                       label=class_name, alpha=0.6, s=50)
    
    plt.legend(fontsize=12)
    plt.title('t-SNE Projection of GNN Node Embeddings', fontsize=16)
    plt.tight_layout()
    plt.savefig('node_embeddings_tsne.png', dpi=300)
    
    # Alternative: PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(test_embeddings)
    
    plt.figure(figsize=(12, 10))
    for i, class_name in enumerate(class_names):
        mask = labels == i
        if np.any(mask):  # Only plot if we have at least one point
            plt.scatter(pca_result[mask, 0], pca_result[mask, 1], 
                       label=class_name, alpha=0.6, s=50)
    
    plt.legend(fontsize=12)
    plt.title(f'PCA Projection of GNN Node Embeddings\nExplained Variance: {sum(pca.explained_variance_ratio_):.2f}', fontsize=16)
    plt.tight_layout()
    plt.savefig('node_embeddings_pca.png', dpi=300)

# Usage example - add this to your main code
fit_indices = np.where(original_index >= len(train_df))[0]  # Get indices of test samples in the dataset
test_labels = fit_df_cleaned[LABEL_COLUMN].map(
    {class_name: i for i, class_name in enumerate(class_names)}).values
visualize_embeddings(model, data, test_labels, class_names, fit_indices)




def feature_importance_analysis(model, data, feature_names):
    # Extract weights from the first layer
    weights = model.conv1.lin.weight.cpu().detach().numpy()
    
    # Calculate feature importance as the mean absolute weight
    importance = np.abs(weights).mean(axis=0)
    
    # Sort features by importance
    sorted_idx = np.argsort(importance)[::-1]
    sorted_importance = importance[sorted_idx]
    sorted_features = [feature_names[i] for i in sorted_idx]
    
    # Plot top 20 features
    plt.figure(figsize=(14, 8))
    plt.barh(range(20), sorted_importance[:20], align='center')
    plt.yticks(range(20), sorted_features[:20])
    plt.xlabel('Mean Absolute Weight')
    plt.title('Top 20 Feature Importance in GNN First Layer')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    #plt.show()

# Usage
feature_importance_analysis(model, data, valid_features)




import networkx as nx

def visualize_knn_graph_sample(edge_index, node_labels, class_names, n_samples=500):
    # Sample a subset of nodes for visualization
    np.random.seed(42)
    sample_indices = np.random.choice(len(node_labels), size=n_samples, replace=False)
    
    # Create a networkx graph
    G = nx.Graph()
    
    # Add sampled nodes
    for i in sample_indices:
        G.add_node(i, label=node_labels[i])
    
    # Add edges between sampled nodes
    edge_array = edge_index.cpu().numpy()
    for i in range(edge_array.shape[1]):
        source, target = edge_array[0, i], edge_array[1, i]
        if source in sample_indices and target in sample_indices:
            G.add_edge(source, target)
    
    # Plot
    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(G, seed=42)
    
    # Color nodes by class
    colors = []
    for node in G.nodes():
        label = G.nodes[node]['label']
        colors.append(label)
    
    nx.draw_networkx(G, pos, node_size=60, node_color=colors, 
                     cmap=plt.cm.tab10, width=0.3, alpha=0.7,
                     with_labels=False)
    
    # Add legend
    handles = []
    for i, class_name in enumerate(class_names):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor=plt.cm.tab10(i), markersize=10,
                                 label=class_name))
    plt.legend(handles=handles, loc='upper right')
    
    plt.title(f'KNN Graph Structure Sample (n={n_samples} nodes)', fontsize=16)
    plt.axis('off')
    plt.savefig('knn_graph_sample.png', dpi=300)
    #plt.show()

# Usage
node_labels = y_all.cpu().numpy()
node_labels = np.array([label if label != -1 else len(class_names) for label in node_labels])
class_names_with_unknown = list(class_names) + ["Unknown"]
visualize_knn_graph_sample(edge_index, node_labels, class_names_with_unknown)



def plot_feature_class_correlations(df, features, class_column):
    # Select a subset of interesting features
    df = df.copy()
    df[class_column] = df[class_column].astype(str)  # Convert to string to avoid endianness issues
    

    selected_features = features[:12] if len(features) > 12 else features
    
    plt.figure(figsize=(18, 12))
    for i, feature in enumerate(selected_features):
        plt.subplot(3, 4, i+1)
        for class_name in df[class_column].unique():
            subset = df[df[class_column] == class_name]
            sns.kdeplot(subset[feature], label=class_name)
        
        plt.title(f'Distribution of {feature}')
        plt.xlabel(feature)
        if i % 4 == 0:
            plt.ylabel('Density')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig('feature_class_distributions.png', dpi=300)
    #plt.show()

# Select key astronomical features
astronomical_features = [
    "parallax", "pmra", "pmdec", "MAD", "true_amplitude", 
    "j_med_mag-ks_med_mag", "h_med_mag-ks_med_mag", "z_med_mag-ks_med_mag",
    "ls_period1", "pdm_period1", "ce_period1"
]

# Filter available features
available_astro_features = [f for f in astronomical_features if f in fit_df_cleaned.columns]
plot_feature_class_correlations(fit_df_cleaned, available_astro_features, LABEL_COLUMN)








def plot_period_amplitude(df, class_column):
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot
    for class_name in df[class_column].unique():
        subset = df[df[class_column] == class_name]
        plt.scatter(subset['true_period'], subset['true_amplitude'], 
                   label=class_name, alpha=0.7, s=50)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Period (days)', fontsize=14)
    plt.ylabel('Amplitude', fontsize=14)
    plt.title('Period-Amplitude Relationship by Class', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('period_amplitude.png', dpi=300)
    #plt.show()

# Check if required columns exist
plot_period_amplitude(fit_df_cleaned, LABEL_COLUMN)






