import sys
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from astropy.io import fits
from sklearn.metrics import classification_report, accuracy_score
import subprocess
from scipy.sparse import csr_matrix
import time

# Import visualization functions
from vis import *

#########################################
# UTILITY FUNCTIONS
#########################################

def get_gpu_count():
    """Auto-detect available GPUs"""
    try:
        result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            return len(result.stdout.decode('utf-8').strip().split('\n'))
        return 0
    except:
        return 0

def load_fits_to_df(path):
    """Load FITS file to DataFrame with robust error handling"""
    print(f"Loading {path}...")
    try:
        with fits.open(path) as hdul:
            # Try primary data first, then extension
            if len(hdul) > 1 and hdul[1].data is not None:
                data = hdul[1].data
            else:
                data = hdul[0].data
            
            if hasattr(data, 'names') and data.names:
                # Structured array
                df = pd.DataFrame()
                for col in data.names:
                    df[col] = data[col]
            else:
                # Regular array
                df = pd.DataFrame(data)
            
            print(f"Loaded {len(df)} samples with {len(df.columns)} features")
            return df
    except Exception as e:
        print(f"Error loading FITS file: {e}")
        raise

def get_feature_list(train, test):
    """Get common features between training and test data"""
    # Enhanced feature sets for variable star classification
    variability = [
        "MAD", "eta", "eta_e", "true_amplitude", "mean_var", "std_nxs", 
        "range_cum_sum", "max_slope", "percent_amp", "stet_k", "roms", 
        "lag_auto", "Cody_M", "AD", "med_BRP", "p_to_p_var"
    ]
    colour = [
        "z_med_mag-ks_med_mag", "y_med_mag-ks_med_mag", 
        "j_med_mag-ks_med_mag", "h_med_mag-ks_med_mag"
    ]
    mags = ["ks_med_mag", "ks_mean_mag", "ks_std_mag", "ks_mad_mag", "j_med_mag", "h_med_mag"]
    period = ["true_period"]
    quality = ["chisq", "uwe"]
    coords = ["l", "b", "parallax", "pmra", "pmdec"]
    faps = ["ls_fap", "pdm_fap", "ce_fap", "gp_fap"]
    
    # Add contrastive learning embeddings (128 dimensions)
    embeddings = [str(i) for i in range(128)]
    
    full_set = variability + colour + mags + period + quality + coords + faps + embeddings
    
    # Get intersection of available features
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    common = train_cols.intersection(test_cols)
    usable = [f for f in full_set if f in common]
    
    print(f"Available features in train: {len(train_cols)}")
    print(f"Available features in test: {len(test_cols)}")
    print(f"Common features: {len(common)}")
    print(f"Using {len(usable)} of {len(full_set)} predefined features")
    
    if len(usable) < 10:
        print("Warning: Very few features available. Consider checking feature names.")
        # Fallback: use any numeric columns
        numeric_cols = []
        for col in common:
            if pd.api.types.is_numeric_dtype(train[col]):
                numeric_cols.append(col)
        
        # Exclude obvious non-feature columns
        exclude = ['source_id', 'id', 'index', 'best_class_name']
        numeric_cols = [col for col in numeric_cols if col not in exclude]
        
        if len(numeric_cols) > len(usable):
            print(f"Using fallback: {len(numeric_cols)} numeric columns")
            usable = numeric_cols[:50]  # Limit to 50 features max
    
    return usable

#########################################
# DATA PREPARATION
#########################################

def preprocess_data(train_df, test_df, features, label_col):
    """Robust data preprocessing for GNN training"""
    print("Preprocessing data...")
    
    # Handle missing label column
    if label_col not in train_df.columns:
        raise ValueError(f"Label column '{label_col}' not found in training data")
    
    # Extract features
    X_train = train_df[features].copy()
    X_test = test_df[features].copy()
    
    print(f"Train shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    
    # Handle missing values and infinities
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    
    # Get statistics from training data only
    train_stats = {
        'mean': X_train.mean(),
        'std': X_train.std(),
        'q01': X_train.quantile(0.01),
        'q99': X_train.quantile(0.99)
    }
    
    # Apply outlier clipping based on training data
    X_train = X_train.clip(lower=train_stats['q01'], upper=train_stats['q99'], axis=1)
    X_test = X_test.clip(lower=train_stats['q01'], upper=train_stats['q99'], axis=1)
    
    # Fill NaN values with training mean
    X_train = X_train.fillna(train_stats['mean'])
    X_test = X_test.fillna(train_stats['mean'])
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    x_train = torch.tensor(X_train_scaled, dtype=torch.float32)
    x_test = torch.tensor(X_test_scaled, dtype=torch.float32)
    
    # Encode labels
    label_encoder = LabelEncoder()
    train_labels = train_df[label_col].fillna('UNKNOWN')  # Handle NaN labels
    y_train = torch.tensor(label_encoder.fit_transform(train_labels), dtype=torch.long)
    
    num_classes = len(label_encoder.classes_)
    class_names = label_encoder.classes_
    
    print(f"Number of classes: {num_classes}")
    print("Class distribution:")
    for i, cls in enumerate(class_names):
        count = (y_train == i).sum().item()
        print(f"  {i}: {cls} - {count} samples ({count/len(y_train)*100:.1f}%)")
    
    return x_train, x_test, y_train, label_encoder, scaler

def build_sparse_graph(features, method='spatial_grid', sparsity=0.01, max_edges=10000000):
    """Build sparse graph for large datasets using spatial/hierarchical methods"""
    print(f"Building sparse graph using {method} method...")
    
    if isinstance(features, torch.Tensor):
        data_np = features.cpu().numpy()
    else:
        data_np = np.array(features)
    
    n_samples, n_features = data_np.shape
    print(f"Processing {n_samples} samples with {n_features} features")
    
    if method == 'spatial_grid':
        return build_spatial_grid_graph(data_np, sparsity, max_edges)
    elif method == 'random_walks':
        return build_random_walk_graph(data_np, sparsity, max_edges)
    elif method == 'hierarchical':
        return build_hierarchical_graph(data_np, sparsity, max_edges)
    else:
        return build_radius_graph(data_np, sparsity, max_edges)

def build_spatial_grid_graph(data_np, sparsity=0.01, max_edges=10000000):
    """Build graph by connecting points in spatial grid cells"""
    from sklearn.decomposition import PCA
    
    n_samples = data_np.shape[0]
    
    # Use PCA to reduce to manageable dimensions for spatial indexing
    if data_np.shape[1] > 8:
        pca = PCA(n_components=8)
        data_reduced = pca.fit_transform(data_np)
    else:
        data_reduced = data_np
    
    # Normalize to [0,1] for grid
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(data_reduced)
    
    # Create spatial grid
    n_dims = data_norm.shape[1]
    grid_size = int(np.power(n_samples * sparsity / 10, 1/n_dims)) + 1
    grid_size = max(2, min(grid_size, 20))  # Reasonable bounds
    
    print(f"Using {grid_size}^{n_dims} spatial grid")
    
    # Assign points to grid cells
    grid_indices = (data_norm * (grid_size - 1)).astype(int)
    
    # Create dictionary of cell -> point indices
    cells = {}
    for i, cell_idx in enumerate(grid_indices):
        cell_key = tuple(cell_idx)
        if cell_key not in cells:
            cells[cell_key] = []
        cells[cell_key].append(i)
    
    # Connect points within same cell and neighboring cells
    edges = []
    
    def get_neighbor_cells(cell_idx, n_dims):
        """Get all neighboring cells (including diagonals)"""
        neighbors = []
        for offset in range(3**n_dims):
            neighbor = list(cell_idx)
            temp = offset
            for dim in range(n_dims):
                delta = (temp % 3) - 1
                neighbor[dim] += delta
                temp //= 3
                if neighbor[dim] < 0 or neighbor[dim] >= grid_size:
                    break
            else:
                neighbors.append(tuple(neighbor))
        return neighbors
    
    edge_count = 0
    for cell_idx, points in cells.items():
        if edge_count >= max_edges:
            break
            
        # Get neighboring cells
        neighbor_cells = get_neighbor_cells(cell_idx, n_dims)
        
        # Collect all points in this cell and neighbors
        all_points = points[:]
        for neighbor_cell in neighbor_cells:
            if neighbor_cell in cells:
                all_points.extend(cells[neighbor_cell])
        
        # Connect points within local neighborhood
        for i, p1 in enumerate(points):
            for p2 in all_points[i+1:]:
                if edge_count < max_edges:
                    edges.append([p1, p2])
                    edges.append([p2, p1])  # Make undirected
                    edge_count += 2
    
    if len(edges) == 0:
        print("Warning: No edges created, using minimal random graph")
        edges = build_minimal_random_graph(n_samples, 1000)
    
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    print(f"Created spatial grid graph with {edge_index.shape[1]} edges")
    print(f"Average degree: {edge_index.shape[1] / n_samples:.1f}")
    
    return edge_index

def build_hierarchical_graph(data_np, sparsity=0.01, max_edges=10000000):
    """Build hierarchical graph using clustering"""
    from sklearn.cluster import MiniBatchKMeans
    
    n_samples = data_np.shape[0]
    
    # Multi-level clustering
    edges = []
    current_data = data_np
    current_indices = np.arange(n_samples)
    
    levels = 3  # Number of hierarchical levels
    for level in range(levels):
        n_clusters = max(10, int(len(current_data) * sparsity * (level + 1)))
        n_clusters = min(n_clusters, len(current_data) // 2)
        
        if n_clusters < 2:
            break
            
        print(f"Level {level}: {len(current_data)} points -> {n_clusters} clusters")
        
        # Cluster current level
        kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1000)
        cluster_labels = kmeans.fit_predict(current_data)
        
        # Connect points within each cluster
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_points = current_indices[cluster_mask]
            
            # Connect all pairs within cluster (make it sparse)
            cluster_points = cluster_points[:min(len(cluster_points), 20)]  # Limit cluster size
            for i, p1 in enumerate(cluster_points):
                for p2 in cluster_points[i+1:]:
                    if len(edges) < max_edges:
                        edges.append([p1, p2])
                        edges.append([p2, p1])
        
        # Prepare for next level - use cluster centers
        if level < levels - 1:
            current_data = kmeans.cluster_centers_
            current_indices = np.arange(len(current_data)) + n_samples + level * 1000
    
    if len(edges) == 0:
        edges = build_minimal_random_graph(n_samples, 1000)
    
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    print(f"Created hierarchical graph with {edge_index.shape[1]} edges")
    print(f"Average degree: {edge_index.shape[1] / n_samples:.1f}")
    
    return edge_index

def build_radius_graph(data_np, sparsity=0.01, max_edges=10000000):
    """Build graph by connecting points within a radius (subsampled)"""
    n_samples = data_np.shape[0]
    
    # Subsample for radius calculation
    sample_size = min(5000, n_samples)
    sample_indices = np.random.choice(n_samples, sample_size, replace=False)
    sample_data = data_np[sample_indices]
    
    # Calculate pairwise distances on sample
    from sklearn.metrics.pairwise import euclidean_distances
    distances = euclidean_distances(sample_data[:1000])  # Even smaller sample for radius
    
    # Determine radius for desired sparsity
    target_connections = int(sparsity * 1000 * 1000)
    radius = np.percentile(distances[distances > 0], 
                          100 * target_connections / (1000 * 1000))
    
    print(f"Using radius {radius:.3f} for graph construction")
    
    # Build graph with radius on subsampled data
    edges = []
    batch_size = 1000
    
    for i in range(0, sample_size, batch_size):
        end_i = min(i + batch_size, sample_size)
        batch_data = sample_data[i:end_i]
        
        for j in range(i, sample_size, batch_size):
            end_j = min(j + batch_size, sample_size)
            if len(edges) >= max_edges:
                break
                
            batch_distances = euclidean_distances(batch_data, sample_data[j:end_j])
            
            # Find connections within radius
            within_radius = np.where(batch_distances <= radius)
            for local_i, local_j in zip(within_radius[0], within_radius[1]):
                global_i = sample_indices[i + local_i]
                global_j = sample_indices[j + local_j]
                if global_i != global_j and len(edges) < max_edges:
                    edges.append([global_i, global_j])
    
    if len(edges) == 0:
        edges = build_minimal_random_graph(n_samples, 1000)
    
    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
    print(f"Created radius graph with {edge_index.shape[1]} edges")
    print(f"Average degree: {edge_index.shape[1] / n_samples:.1f}")
    
    return edge_index

def build_minimal_random_graph(n_samples, min_edges=1000):
    """Build minimal random graph as fallback"""
    print("Building minimal random graph as fallback...")
    edges = []
    
    # Create a few random connections per node
    connections_per_node = max(2, min_edges // n_samples)
    
    for i in range(n_samples):
        # Random connections
        targets = np.random.choice(n_samples, 
                                 size=min(connections_per_node, n_samples-1), 
                                 replace=False)
        targets = targets[targets != i]  # Remove self
        
        for target in targets:
            edges.append([i, target])
    
    return edges

#########################################
# MODEL DEFINITION
#########################################

class ImprovedGCN(torch.nn.Module):
    """Enhanced Graph Convolutional Network with better architecture"""
    
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5, num_layers=3):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Output layer
        self.convs.append(GCNConv(hidden_channels, out_channels))
        
        # Dropout and normalization
        self.dropout_layer = Dropout(dropout)
        
        # Skip connections
        self.skip_connections = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.skip_connections.append(Linear(hidden_channels, hidden_channels))

    def forward(self, x, edge_index):
        # Input layer
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = self.dropout_layer(x)
        
        # Hidden layers with skip connections
        for i in range(1, self.num_layers - 1):
            identity = x
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = self.dropout_layer(x)
            
            # Skip connection
            if i > 0:
                x = x + self.skip_connections[i-1](identity)
        
        # Output layer (no activation)
        x = self.convs[-1](x, edge_index)
        return x

    def get_embeddings(self, x, edge_index):
        """Extract embeddings from penultimate layer"""
        with torch.no_grad():
            # Forward through all but last layer
            for i in range(len(self.convs) - 1):
                x = self.convs[i](x, edge_index)
                x = F.relu(x)
                if i < len(self.convs) - 2:  # Don't apply dropout to final embedding
                    x = self.dropout_layer(x)
            return x

#########################################
# TRAINING FUNCTIONS
#########################################

def create_balanced_sampler(y_train):
    """Create a balanced sampler for imbalanced datasets"""
    class_counts = torch.bincount(y_train)
    class_weights = 1.0 / class_counts.float()
    
    # Normalize weights
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    
    # Create sample weights
    sample_weights = class_weights[y_train]
    
    return sample_weights, class_weights

def train_model_robust(model, data, optimizer, criterion, device, epochs=200, patience=30):
    """Robust training with early stopping and monitoring"""
    print("\nStarting training...")
    
    model = model.to(device)
    data = data.to(device)
    
    # Split training data for validation
    n_train = len(data.y)
    train_mask = torch.zeros(n_train, dtype=torch.bool)
    val_mask = torch.zeros(n_train, dtype=torch.bool)
    
    # Stratified split
    for class_idx in torch.unique(data.y):
        class_indices = (data.y == class_idx).nonzero().flatten()
        n_class = len(class_indices)
        n_train_class = max(1, int(0.8 * n_class))
        
        train_indices = class_indices[:n_train_class]
        val_indices = class_indices[n_train_class:]
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
    
    print(f"Training samples: {train_mask.sum()}, Validation samples: {val_mask.sum()}")
    
    best_val_acc = 0
    best_model_state = None
    patience_counter = 0
    losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            
            # Training accuracy
            train_pred = out[train_mask].argmax(dim=1)
            train_acc = (train_pred == data.y[train_mask]).float().mean()
            
            # Validation accuracy
            val_pred = out[val_mask].argmax(dim=1)
            val_acc = (val_pred == data.y[val_mask]).float().mean()
            
            losses.append(loss.item())
            train_accs.append(train_acc.item())
            val_accs.append(val_acc.item())
        
        # Early stopping check
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print progress
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d}: Loss={loss:.4f}, Train Acc={train_acc:.3f}, Val Acc={val_acc:.3f}")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best validation accuracy: {best_val_acc:.3f}")
    
    return model, losses, train_accs, val_accs

#########################################
# MAIN WORKFLOW FUNCTION
#########################################

def train_gnn(train_df, test_df, features, label_col, out_file, k_neighbors=10, epochs=200):
    """Complete GNN workflow with robust error handling"""
    print("\n=== Enhanced GNN Training Workflow ===")
    
    # Setup
    os.makedirs('figures', exist_ok=True)
    start_time = time.time()
    
    # Device setup
    num_gpus = get_gpu_count()
    if num_gpus > 0 and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA with {num_gpus} GPUs")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    try:
        # Data preparation
        x_train, x_test, y_train, label_encoder, scaler = preprocess_data(
            train_df, test_df, features, label_col
        )
        
        # Graph construction for training data
        print("\nBuilding training graph...")
        edge_index_train = build_knn_graph_sklearn(x_train, k=k_neighbors)
        
        # Create training data object
        train_data = Data(x=x_train, edge_index=edge_index_train, y=y_train)
        
        # Model setup
        in_channels = x_train.shape[1]
        num_classes = len(label_encoder.classes_)
        hidden_channels = min(128, max(64, in_channels // 2))
        
        print(f"Model architecture: {in_channels} -> {hidden_channels} -> {num_classes}")
        
        model = ImprovedGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            dropout=0.3,
            num_layers=3
        )
        
        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        # Balanced loss function
        sample_weights, class_weights = create_balanced_sampler(y_train)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
        
        # Training
        model, losses, train_accs, val_accs = train_model_robust(
            model, train_data, optimizer, criterion, device, epochs=epochs
        )
        
        # Inference on test data
        print("\nRunning inference...")
        
        # Build combined graph for inference
        x_combined = torch.cat([x_train, x_test], dim=0)
        edge_index_combined = build_knn_graph_sklearn(x_combined, k=k_neighbors)
        
        # Create combined data object
        combined_data = Data(x=x_combined, edge_index=edge_index_combined)
        combined_data = combined_data.to(device)
        
        # Run inference
        model.eval()
        with torch.no_grad():
            out = model(combined_data.x, combined_data.edge_index)
            probs = F.softmax(out, dim=1)
            
            # Extract test predictions
            test_start_idx = len(x_train)
            test_probs = probs[test_start_idx:].cpu().numpy()
            test_preds = out[test_start_idx:].argmax(dim=1).cpu().numpy()
            test_confs = test_probs.max(axis=1)
            
            # Get embeddings for visualization
            embeddings = model.get_embeddings(combined_data.x, combined_data.edge_index)
            test_embeddings = embeddings[test_start_idx:].cpu().numpy()
        
        # Decode predictions
        pred_labels = label_encoder.inverse_transform(test_preds)
        
        # Save results
        test_df_result = test_df.copy()
        test_df_result['gnn_predicted_class_id'] = test_preds
        test_df_result['gnn_predicted_class'] = pred_labels
        test_df_result['gnn_confidence'] = test_confs
        
        test_df_result.to_csv(out_file, index=False)
        print(f"Saved predictions to {out_file}")
        
        # Evaluation if ground truth available
        if label_col in test_df.columns:
            y_true = test_df[label_col].fillna('UNKNOWN')
            print("\nTest set evaluation:")
            print(classification_report(y_true, pred_labels))
            
            accuracy = accuracy_score(y_true, pred_labels)
            print(f"Test Accuracy: {accuracy:.3f}")
        
        # Create visualizations
        print("\nGenerating visualizations...")
        try:
            create_gnn_dashboard(
                test_df_result, label_col, features, model, test_preds, test_probs,
                label_encoder.classes_, test_embeddings, losses, edge_index_combined
            )
        except Exception as e:
            print(f"Warning: Visualization error: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"\nGNN training completed in {elapsed_time:.1f} seconds")
        
        return model, test_df_result
        
    except Exception as e:
        print(f"Error in GNN training: {e}")
        import traceback
        traceback.print_exc()
        raise

#########################################
# ENTRY POINT
#########################################

def main():
    """Main entry point with robust argument parsing"""
    # Parse arguments with better defaults
    if len(sys.argv) >= 3:
        train_path = sys.argv[1]
        test_path = sys.argv[2]
        out_file = sys.argv[3] if len(sys.argv) > 3 else "gnn_predictions.csv"
    else:
        # Default paths
        train_path = "../PRIMVS/PRIMVS_P_GAIA.fits"
        test_path = "../PRIMVS/PRIMVS_P.fits"
        out_file = "gnn_predictions.csv"
        print("Using default file paths")
        print(f"Train: {train_path}")
        print(f"Test: {test_path}")
        print(f"Output: {out_file}")
    
    try:
        # Load data
        train_df = load_fits_to_df(train_path)
        test_df = load_fits_to_df(test_path)
        
        # Configuration
        label_col = "best_class_name"
        
        # Check if label column exists
        if label_col not in train_df.columns:
            available_cols = [col for col in train_df.columns if 'class' in col.lower() or 'label' in col.lower()]
            if available_cols:
                label_col = available_cols[0]
                print(f"Using alternative label column: {label_col}")
            else:
                raise ValueError(f"No suitable label column found. Available columns: {list(train_df.columns)}")
        
        # Show class distribution
        print(f"\nUsing label column: {label_col}")
        print("Class distribution in training data:")
        print(train_df[label_col].value_counts())
        
        # Get features
        features = get_feature_list(train_df, test_df)
        
        if len(features) == 0:
            raise ValueError("No suitable features found for training")
        
        # Train GNN
        model, results = train_gnn(
            train_df, test_df, features, label_col, out_file,
            k_neighbors=8,  # Reduced for stability
            epochs=150
        )
        
        print("\n=== GNN Training Complete ===")
        print(f"Results saved to: {out_file}")
        print(f"Visualizations saved to: figures/")
        
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        print("Please check file paths and try again.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()