import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from astropy.io import fits
from sklearn.metrics import classification_report
import faiss
import subprocess

# Import visualization functions
from vis import *

#########################################
# UTILITY FUNCTIONS
#########################################

def get_gpu_count():
    """Auto-detect available GPUs"""
    try:
        result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE)
        return len(result.stdout.decode('utf-8').strip().split('\n'))
    except:
        return 0

def load_fits_to_df(path):
    """Load FITS file to DataFrame with error handling"""
    print(f"Loading {path}...")
    with fits.open(path) as hdul:
        try:
            return pd.DataFrame(hdul[1].data)
        except:
            # Handle structured arrays
            data = hdul[1].data
            columns = data.names
            df = pd.DataFrame()
            for col in columns:
                df[col] = data[col]
            return df

def get_feature_list(train, test):
    """Get common features between training and test data"""
    # Basic feature sets for variable star classification
    variability = ["MAD", "eta", "eta_e", "true_amplitude", "mean_var", "std_nxs", 
                 "range_cum_sum", "max_slope", "percent_amp", "stet_k", "roms", 
                 "lag_auto", "Cody_M", "AD"]
    colour = ["z_med_mag-ks_med_mag", "y_med_mag-ks_med_mag", "j_med_mag-ks_med_mag", 
            "h_med_mag-ks_med_mag"]
    mags = ["ks_med_mag", "ks_std_mag", "ks_mad_mag"]
    period = ["true_period"]
    quality = ["chisq", "uwe"]
    coords = ["l", "b", "parallax", "pmra", "pmdec"]
    faps = ["ls_fap", "pdm_fap", "ce_fap", "gp_fap"]
    
    # Add contrastive learning embeddings (128 dimensions)
    embeddings = [str(i) for i in range(128)]
    
    full_set = variability + colour + mags + period + quality + coords + faps + embeddings
    common = set(train.columns).intersection(test.columns)
    usable = [f for f in full_set if f in common]
    print(f"Using {len(usable)} of {len(full_set)} total features")
    return usable

#########################################
# DATA PREPARATION
#########################################

def preprocess_data(train_df, test_df, features, label_col):
    """Process data for GNN training"""
    print("Preprocessing data...")
    
    # Handle missing values and infinities
    X_train = train_df[features].replace([np.inf, -np.inf], np.nan)
    X_test = test_df[features].replace([np.inf, -np.inf], np.nan)
    
    # Get statistics from training data
    train_quantiles = X_train.quantile([0.001, 0.999])
    train_min = train_quantiles.iloc[0]
    train_max = train_quantiles.iloc[1]
    train_mean = X_train.mean()
    
    # Apply clipping and imputation
    X_train = X_train.clip(lower=train_min, upper=train_max, axis=1)
    X_train.fillna(train_mean, inplace=True)
    
    X_test = X_test.clip(lower=train_min, upper=train_max, axis=1)
    X_test.fillna(train_mean, inplace=True)
    
    # Standardize data
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
    print("Class distribution:")
    for i, cls in enumerate(label_encoder.classes_):
        count = (y_train == i).sum().item()
        print(f"  {i}: {cls} - {count} samples")
    
    return x_train, x_test, y_train, label_encoder, train_df, test_df

def build_knn_graph(features, k=8, use_gpu=True):
    """Build KNN graph for GNN using FAISS"""
    if isinstance(features, torch.Tensor):
        data_np = features.cpu().numpy().astype('float32')
    else:
        data_np = features.astype('float32')
    
    n, d = data_np.shape
    print(f"Building KNN graph for {n} points with dimension {d}")
    
    # Initialize FAISS index with GPU if available
    gpu_available = use_gpu and faiss.get_num_gpus() > 0
    if gpu_available:
        print(f"Using GPU for FAISS (found {faiss.get_num_gpus()} GPUs)")
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
    else:
        print("Using CPU for FAISS")
        index = faiss.IndexFlatL2(d)
    
    # Add data to index
    index.add(data_np)
    
    # Process in batches
    batch_size = 500000
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

#########################################
# MODEL DEFINITION
#########################################

class GCN(torch.nn.Module):
    """Graph Convolutional Network for star classification"""
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        """Extract embeddings from penultimate layer"""
        with torch.no_grad():
            return self.conv1(x, edge_index).detach()

#########################################
# TRAINING FUNCTIONS
#########################################

def train_model(model, data, y_train, num_classes, device, batch_size=4096, epochs=150):
    """Train the GNN model"""
    print("\nTraining model...")
    
    # Compute class weights for imbalanced classes
    class_counts = torch.bincount(y_train)
    class_weights = 1.0 / (class_counts.float() + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = class_weights.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-5
    )
    
    # Training loop with batching
    model.train()
    best_loss = float('inf')
    patience_counter = 0
    patience = 20  # Early stopping patience
    losses = []
    best_model = None
    
    for epoch in range(epochs):
        epoch_loss = 0
        
        # Shuffle indices
        train_indices = torch.randperm(len(y_train))
        num_batches = (len(train_indices) + batch_size - 1) // batch_size
        
        for i in range(num_batches):
            batch_indices = train_indices[i*batch_size:(i+1)*batch_size]
            optimizer.zero_grad()
            
            # Forward pass
            out = model(data.x, data.edge_index)
            loss = F.cross_entropy(out[batch_indices], data.y[batch_indices], weight=class_weights)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_loss)
        
        # Print progress
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d}, Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Check early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                # Save best model
                best_model = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                    break
    
    # Load best model
    if best_model is not None:
        model.load_state_dict(best_model)
    
    return model, losses

#########################################
# INFERENCE FUNCTIONS
#########################################

def run_inference(model, combined_data, edge_index, x_train_len, device):
    """Run inference on test data"""
    print("\nRunning inference on test data...")
    
    model.eval()
    with torch.no_grad():
        combined_data = combined_data.to(device)
        edge_index = edge_index.to(device)
        
        # Forward pass
        logits = model(combined_data, edge_index)
        probs = F.softmax(logits, dim=1)
        
        # Extract predictions for test data
        test_indices = torch.arange(x_train_len, len(combined_data)).to(device)
        test_logits = logits[test_indices]
        test_probs = probs[test_indices]
        
        # Get predictions
        preds = test_logits.argmax(dim=1).cpu().numpy()
        confs = test_probs.max(dim=1)[0].cpu().numpy()
        all_probs = test_probs.cpu().numpy()
        
        # Generate embeddings for visualization
        embeddings = model.get_embeddings(combined_data, edge_index)
        test_embeddings = embeddings[test_indices].cpu().numpy()
    
    return preds, confs, all_probs, test_embeddings

#########################################
# MAIN WORKFLOW FUNCTION
#########################################

def train_gnn(train_df, test_df, features, label_col, out_file, k_neighbors=8, batch_size=4096, epochs=150):
    """Complete GNN workflow: preprocessing, training, inference, visualization"""
    print("\n=== Enhanced GNN Training Workflow ===")
    
    # Set up directories
    os.makedirs('figures', exist_ok=True)
    os.makedirs(os.path.dirname(out_file) if os.path.dirname(out_file) else '.', exist_ok=True)
    
    # Determine device
    num_gpus = get_gpu_count()
    if num_gpus > 0:
        device = torch.device('cuda')
        print(f"Using CUDA with {num_gpus} GPUs")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    # === DATA PREPARATION ===
    x_train, x_test, y_train, label_encoder, train_df, test_df = preprocess_data(
        train_df, test_df, features, label_col
    )
    
    # === GRAPH CONSTRUCTION ===
    print("\nConstructing KNN graph...")
    edge_index_train = build_knn_graph(x_train, k=k_neighbors, use_gpu=(device.type == 'cuda'))
    
    # Create training data object
    data = Data(x=x_train, edge_index=edge_index_train, y=y_train)
    data = data.to(device)
    
    # === MODEL INITIALIZATION ===
    # Set hidden dimension based on data complexity
    in_channels = x_train.shape[1]
    num_classes = len(label_encoder.classes_)
    hidden_dim = min(128, max(64, num_classes * 4))
    print(f"Using hidden dimension of {hidden_dim}")
    
    model = GCN(
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        out_channels=num_classes,
        dropout=0.5  # Increased dropout for regularization
    ).to(device)
    
    # === TRAINING ===
    model, losses = train_model(
        model, data, y_train, num_classes, device, 
        batch_size=batch_size, epochs=epochs
    )
    
    # Plot training loss
    plot_gnn_training_loss(losses)
    
    # === INFERENCE ===
    # Build combined graph (train + test) for inference
    combined_data = torch.cat([x_train, x_test], dim=0)
    edge_index_combined = build_knn_graph(
        combined_data, k=k_neighbors, use_gpu=(device.type == 'cuda')
    )
    
    # Run inference
    preds, confs, all_probs, test_embeddings = run_inference(
        model, combined_data, edge_index_combined, len(x_train), device
    )
    
    # Decode predictions
    pred_labels = label_encoder.inverse_transform(preds)
    
    # === SAVE RESULTS ===
    test_df_result = test_df.copy()
    test_df_result['gnn_predicted_class_id'] = preds
    test_df_result['gnn_predicted_class'] = pred_labels
    test_df_result['gnn_confidence'] = confs
    
    test_df_result.to_csv(out_file, index=False)
    print(f"Saved predictions to {out_file}")
    
    # === VISUALIZATIONS ===
    print("\nCreating visualizations...")
    
    # Create dashboard
    create_gnn_dashboard(
        test_df_result, label_col, features, model, preds, all_probs,
        label_encoder.classes_, test_embeddings, losses, edge_index_combined
    )
    
    # Evaluate if ground truth is available
    if label_col in test_df.columns:
        print("\nEvaluation on test set:")
        y_true = test_df[label_col]
        
        # Print classification metrics
        print(classification_report(y_true, pred_labels))
    
    return model, test_df_result

#########################################
# ENTRY POINT
#########################################

def main():
    # Parse command-line arguments or use defaults
    if len(sys.argv) < 3:
        train_path = "../PRIMVS/PRIMVS_P_GAIA.fits"
        test_path = "../PRIMVS/PRIMVS_P.fits"
        out_file = "gnn_predictions.csv"
        print("Using default file paths")
    else:
        train_path, test_path = sys.argv[1:3]
        out_file = sys.argv[3] if len(sys.argv) > 3 else "gnn_predictions.csv"
    
    # Load data
    train_df = load_fits_to_df(train_path)
    test_df = load_fits_to_df(test_path)
    
    # Get label column and print class distribution
    label_col = "best_class_name"
    print("\nClass distribution in training data:")
    print(train_df[label_col].value_counts())
    
    # Get features
    features = get_feature_list(train_df, test_df)
    
    # Train GNN model
    model, _ = train_gnn(train_df, test_df, features, label_col, out_file)
    
    print("\nGNN processing complete!")

if __name__ == "__main__":
    main()
