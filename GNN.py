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
from astropy.io import fits
from sklearn.metrics import classification_report, confusion_matrix
import faiss  # Add FAISS for fast KNN
from vis import *

# === Configuration ===
class Config:
    # Feature definitions
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
    
    # Model parameters
    K_NEIGHBORS = 8
    MAX_CLASSES = 15
    BATCH_SIZE = 4096
    USE_CPU = False
    LABEL_COLUMN = "best_class_name"
    EPOCHS = 150

# === Data Loading Functions ===
def load_fits_to_dataframe(fits_path):
    """Load a FITS file into a pandas DataFrame"""
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

# === Data Preprocessing Functions ===
def preprocess_data(train_df, test_df, feature_cols, label_col):
    """Preprocess data for model training"""
    # Identify common features
    train_features = set(train_df.columns)
    test_features = set(test_df.columns)
    common_features = list(train_features.intersection(test_features))
    available_features = [f for f in feature_cols if f in common_features]
    
    print(f"Found {len(available_features)} common features across both datasets")
    if len(available_features) < 10:
        print("WARNING: Very few features available. Identifying alternative features...")
        potential_embeddings = [col for col in common_features if (
            col.startswith('z') or col.startswith('h') or col.startswith('u')
        ) and (
            'pca' in col.lower() or 'tsne' in col.lower() or 'umap' in col.lower()
        )]
        
        if potential_embeddings:
            print(f"Found {len(potential_embeddings)} potential embedding features")
            available_features += [col for col in potential_embeddings if col not in available_features]
    
    print(f"Using {len(available_features)} features for model training")
    
    # Process feature matrices
    X_train = train_df[available_features].copy()
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"NaN values in training data: {X_train.isna().sum().sum()}")
    
    X_test = test_df[available_features].copy()
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"NaN values in testing data: {X_test.isna().sum().sum()}")
    
    # Compute statistics from training data only
    train_min = X_train.quantile(0.001)
    train_max = X_train.quantile(0.999)
    train_mean = X_train.mean()
    
    # Apply clipping and imputation
    X_train = X_train.clip(lower=train_min, upper=train_max, axis=1)
    X_train.fillna(train_mean, inplace=True)
    
    X_test = X_test.clip(lower=train_min, upper=train_max, axis=1)
    X_test.fillna(train_mean, inplace=True)
    
    # Standardize
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_combined)
    
    # Split back to train and test
    X_train_scaled = X_scaled[:len(X_train)]
    X_test_scaled = X_scaled[len(X_train):]
    
    # Convert to tensors
    x_train = torch.tensor(X_train_scaled, dtype=torch.float)
    x_test = torch.tensor(X_test_scaled, dtype=torch.float)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train = torch.tensor(label_encoder.fit_transform(train_df[label_col]), dtype=torch.long)
    num_classes = len(label_encoder.classes_)
    
    print(f"Using {num_classes} classes for training")
    print("Class mapping:")
    for i, cls in enumerate(label_encoder.classes_):
        print(f"  {i}: {cls}")
    
    return x_train, x_test, y_train, label_encoder, available_features

# === Graph Construction ===
def build_knn_graph_faiss(data, k, use_gpu=True):
    """Fast KNN graph construction using FAISS"""
    if isinstance(data, torch.Tensor):
        data_np = data.cpu().numpy().astype('float32')
    else:
        data_np = data.astype('float32')
    
    n, d = data_np.shape
    print(f"Building KNN graph for {n} points with dimension {d}")
    
    # Initialize FAISS index with GPU if available
    if use_gpu and faiss.get_num_gpus() > 0:
        print(f"Using GPU for FAISS (found {faiss.get_num_gpus()} GPUs)")
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
    else:
        print("Using CPU for FAISS")
        index = faiss.IndexFlatL2(d)
    
    # For very large datasets, process in batches
    batch_size = 500000  # Adjust based on GPU memory
    
    # Add all data to the index (this is fast)
    index.add(data_np)
    
    # Process queries in batches (this is the slow part)
    all_rows = []
    all_cols = []
    
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        print(f"Processing batch {i//batch_size + 1}/{(n+batch_size-1)//batch_size}")
        batch = data_np[i:end]
        _, indices = index.search(batch, k)
        
        # Create edges
        batch_rows = np.repeat(np.arange(i, end), k)
        batch_cols = indices.reshape(-1)
        
        all_rows.append(batch_rows)
        all_cols.append(batch_cols)
    
    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    
    # Create edge_index tensor
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    
    print(f"Created KNN graph with {edge_index.shape[1]} edges")
    return edge_index

# === Model Definition ===
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
        return self.conv1(x, edge_index).detach()

# === Training Function ===
def train_model(model, data, y_train, num_classes, device, config):
    """Train the GNN model"""
    print(f"Training model with {num_classes} classes...")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # Compute class weights for imbalanced classes
    class_counts = np.bincount(y_train.cpu().numpy())
    class_weights = 1.0 / (class_counts + 1e-10)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    # Batched training
    model.train()
    for epoch in range(config.EPOCHS):
        total_loss = 0
        # Shuffle indices for better training
        train_indices = torch.randperm(len(y_train))
        num_batches = (len(train_indices) + config.BATCH_SIZE - 1) // config.BATCH_SIZE
        
        for i in range(num_batches):
            batch_indices = train_indices[i*config.BATCH_SIZE:(i+1)*config.BATCH_SIZE]
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[batch_indices], data.y[batch_indices], weight=class_weights)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d}, Avg Loss: {total_loss/num_batches:.4f}")
    
    return model

# === Inference Function ===
def run_inference(model, combined_data, edge_index, x_train_len, device):
    """Run inference on test data"""
    print("Running inference on test data...")
    model.eval()
    with torch.no_grad():
        test_data = Data(x=combined_data.to(device), edge_index=edge_index.to(device))
        logits = model(test_data.x, test_data.edge_index)
        probs = F.softmax(logits, dim=1)
        
        # Extract predictions for test data
        test_indices = torch.arange(x_train_len, len(combined_data))
        test_logits = logits[test_indices]
        test_probs = probs[test_indices]
        
        preds = test_logits.argmax(dim=1).cpu().numpy()
        confs = test_probs.max(dim=1)[0].cpu().numpy()
    
    return preds, confs

# === Main Function ===
def main():
    # Parse command-line arguments
    if len(sys.argv) < 3:
        print("Usage: python GNN.py <training_fits_file> <testing_fits_file> [output_file]")
        print("Example: python GNN.py primvs.fits primvs_gaia.fits output_predictions.csv")
        sys.exit(1)
    
    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "fit_with_predictions.csv"
    
    # Initialize configuration
    config = Config()
    
    # Load data
    train_df = load_fits_to_dataframe(training_file)
    print(f"Loaded training data: {len(train_df)} rows")
    
    test_df = load_fits_to_dataframe(testing_file)
    print(f"Loaded testing data: {len(test_df)} rows")
    
    # Analyze classes
    print("\nExamining true_class distribution in training data:")
    train_classes = train_df[config.LABEL_COLUMN].value_counts()
    print(f"Found {len(train_classes)} unique classes")
    print(train_classes.head(10))
    
    # Preprocess data
    x_train, x_test, y_train, label_encoder, available_features = preprocess_data(
        train_df, test_df, config.FEATURE_COLUMNS, config.LABEL_COLUMN
    )
    
    # Setup device
    if config.USE_CPU:
        device = torch.device('cpu')
        print("Using CPU for computation (forced)")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    
    # Build KNN graph for training data
    edge_index_train = build_knn_graph_faiss(
        x_train, 
        k=config.K_NEIGHBORS, 
        use_gpu=(device.type == 'cuda')
    )
    
    # Create training data object
    data = Data(x=x_train, edge_index=edge_index_train, y=y_train)
    data = data.to(device)
    
    # Initialize and train model
    num_classes = len(label_encoder.classes_)
    hidden_dim = min(64, max(32, num_classes * 2))
    model = GCN(in_channels=x_train.shape[1], hidden_channels=hidden_dim, out_channels=num_classes).to(device)
    model = train_model(model, data, y_train, num_classes, device, config)
    
    # Build KNN graph for combined data (train + test)
    combined_data = torch.cat([x_train, x_test], dim=0)
    edge_index_combined = build_knn_graph_faiss(
        combined_data, 
        k=config.K_NEIGHBORS, 
        use_gpu=(device.type == 'cuda')
    )
    
    # Run inference
    preds, confs = run_inference(model, combined_data, edge_index_combined, len(x_train), device)
    
    # Add predictions to test DataFrame
    test_df_result = test_df.copy()
    test_df_result["gnn_predicted_class_id"] = preds
    test_df_result["gnn_predicted_class"] = label_encoder.inverse_transform(preds)
    test_df_result["gnn_confidence"] = confs
    
    # Save results
    test_df_result.to_csv(output_file, index=False)
    print(f"\nSaved predictions to {output_file}")
    
    # Evaluate if ground truth is available
    if config.LABEL_COLUMN in test_df_result.columns:
        print("\nEvaluation on test set:")
        y_true = test_df_result[config.LABEL_COLUMN]
        y_pred = test_df_result["gnn_predicted_class"]
        
        # Print classification metrics
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

if __name__ == "__main__":
    main()