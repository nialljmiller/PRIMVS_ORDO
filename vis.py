import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
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
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import zoom, gaussian_filter
import pandas as pd
import os



def plot_xgb_training_loss(evals_result, save_path='figures/training_loss.png'):
    """Simple function to plot XGBoost training and validation loss"""
    import matplotlib.pyplot as plt
    
    # Extract loss values - handles XGBoost's nested dictionary structure
    train_loss = list(evals_result['train'].values())[0]
    val_loss = list(evals_result['validation'].values())[0]
    iterations = range(1, len(train_loss) + 1)
    
    # Create simple plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_loss, 'b-', label='Training Loss')
    plt.plot(iterations, val_loss, 'r-', label='Validation Loss')
    
    # Mark best model
    best_iter = val_loss.index(min(val_loss)) + 1
    plt.axvline(x=best_iter, color='gray', linestyle='--')
    plt.scatter(best_iter, min(val_loss), color='red', s=100)
    plt.annotate(f'Best: {min(val_loss):.4f} (iter {best_iter})', 
                 xy=(best_iter, min(val_loss)),
                 xytext=(best_iter + 5, min(val_loss)),
                 arrowprops=dict(arrowstyle='->'))
    
    # Basic formatting
    plt.grid(True)
    plt.xlabel('Iterations')
    plt.ylabel('Log Loss')
    plt.title('XGBoost Training Progress')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved loss plot to {save_path}")
    
    return plt




# Set global figure parameters for better readability in publications
def set_plot_style(large_text=False):
    """Set the plot style for publication-quality figures"""
    plt.rcParams['figure.figsize'] = [8, 6]  # Default figure size
    plt.rcParams['figure.dpi'] = 100  # Default figure DPI
    plt.rcParams['savefig.dpi'] = 300  # DPI for saved figures
    
    # Text size settings
    base_size = 14
    if large_text:
        base_size = 18
        
    plt.rcParams['font.size'] = base_size
    plt.rcParams['axes.labelsize'] = base_size + 2  # Axis labels
    plt.rcParams['axes.titlesize'] = base_size + 4  # Axis title
    plt.rcParams['xtick.labelsize'] = base_size + 2  # X-axis tick labels
    plt.rcParams['ytick.labelsize'] = base_size + 2  # Y-axis tick labels
    plt.rcParams['legend.fontsize'] = base_size - 2  # Legend
    plt.rcParams['axes.linewidth'] = 1.5  # Axis line thickness
    
    # Line widths for plots
    plt.rcParams['lines.linewidth'] = 2
    plt.rcParams['lines.markersize'] = 8  # Marker size for plot markers
    
    # Colors and grid
    plt.rcParams['axes.facecolor'] = 'white'  # Axes background color
    plt.rcParams['axes.edgecolor'] = 'black'  # Axes edge color
    plt.rcParams['axes.grid'] = False  # Disable grid by default
    plt.rcParams['grid.alpha'] = 0.5  # Grid line transparency
    plt.rcParams['grid.color'] = "grey"  # Grid line color


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
    plt.savefig('figures/confusion_matrix.png', dpi=300)
    #plt.show()
    
    # Print classification report
    print(classification_report(y_true, y_pred, target_names=class_names))


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
    plt.savefig('figures/knn_graph_sample.png', dpi=300)
    #plt.show()


def plot_period_amplitude(df, class_column, min_confidence=0.7):
    """
    Create a Bailey diagram showing the relationship between period and amplitude for variable stars.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    class_column : str
        The column name containing class labels
    min_confidence : float, optional
        Minimum confidence threshold to include in plot, default=0.7
    """
    plt.figure(figsize=(12, 10))
    
    df = df.copy()
    df[class_column] = df[class_column].astype(str)
    
    # Apply confidence threshold if the column exists
    confidence_col = class_column.replace('predicted_class', 'confidence')
    if confidence_col in df.columns:
        df = df[df[confidence_col] >= min_confidence]
    
    # Create a colormap for the different classes
    unique_classes = df[class_column].unique()
    cmap = plt.cm.get_cmap('tab20', len(unique_classes))
    
    # Create scatter plot
    for i, class_name in enumerate(unique_classes):
        subset = df[df[class_column] == class_name]
        plt.scatter(subset['true_period'], subset['true_amplitude'], 
                   label=class_name, alpha=0.7, s=15, c=[cmap(i)])
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Period (days)', fontsize=14)
    plt.ylabel('Amplitude (mag)', fontsize=14)
    plt.title('Bailey Diagram: Period-Amplitude Relationship by Class', fontsize=16)
    
    # Add grid with custom styling
    plt.grid(alpha=0.3, linestyle='--')
    
    # Optimize legend for many classes
    if len(unique_classes) > 10:
        plt.legend(fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(1, 1))
    else:
        plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('figures/bailey_diagram.png', dpi=300)
    #plt.show()


def plot_hr_diagram(df, class_column, color_index_col=None, magnitude_col=None, min_confidence=0.7):
    """
    Create a Hertzsprung-Russell diagram using color index and magnitude.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    class_column : str
        The column name containing class labels
    color_index_col : str, optional
        The column name for color index (e.g., 'j_med_mag-ks_med_mag')
    magnitude_col : str, optional
        The column name for magnitude (e.g., 'ks_med_mag')
    min_confidence : float, optional
        Minimum confidence threshold to include in plot, default=0.7
    """
    # Determine appropriate columns if not specified
    if color_index_col is None:
        # Try to find a suitable color index column
        color_candidates = [
            'j_med_mag-ks_med_mag',
            'h_med_mag-ks_med_mag',
            'z_med_mag-ks_med_mag',
            'y_med_mag-ks_med_mag'
        ]
        for col in color_candidates:
            if col in df.columns:
                color_index_col = col
                break
        if color_index_col is None:
            raise ValueError("No suitable color index column found. Please specify color_index_col.")
    
    if magnitude_col is None:
        # Try to find a suitable magnitude column
        mag_candidates = ['ks_med_mag', 'j_med_mag', 'h_med_mag']
        for col in mag_candidates:
            if col in df.columns:
                magnitude_col = col
                break
        if magnitude_col is None:
            raise ValueError("No suitable magnitude column found. Please specify magnitude_col.")
    
    plt.figure(figsize=(12, 10))
    
    df = df.copy()
    df[class_column] = df[class_column].astype(str)
    
    # Apply confidence threshold if the column exists
    confidence_col = class_column.replace('predicted_class', 'confidence')
    if confidence_col in df.columns:
        df = df[df[confidence_col] >= min_confidence]
    
    # Remove outliers for better visualization
    q1_color = df[color_index_col].quantile(0.01)
    q3_color = df[color_index_col].quantile(0.99)
    q1_mag = df[magnitude_col].quantile(0.01)
    q3_mag = df[magnitude_col].quantile(0.99)
    
    df_filtered = df[(df[color_index_col] >= q1_color) & 
                    (df[color_index_col] <= q3_color) &
                    (df[magnitude_col] >= q1_mag) & 
                    (df[magnitude_col] <= q3_mag)]
    
    # Create a colormap for the different classes
    unique_classes = df_filtered[class_column].unique()
    cmap = plt.cm.get_cmap('tab20', len(unique_classes))
    
    # Create scatter plot
    for i, class_name in enumerate(unique_classes):
        subset = df_filtered[df_filtered[class_column] == class_name]
        plt.scatter(subset[color_index_col], subset[magnitude_col], 
                   label=class_name, alpha=0.7, s=15, c=[cmap(i)])
    
    # In HR diagrams, magnitude is plotted with decreasing values (brighter stars at top)
    plt.gca().invert_yaxis()
    
    plt.xlabel(f'Color Index ({color_index_col})', fontsize=14)
    plt.ylabel(f'Magnitude ({magnitude_col})', fontsize=14)
    plt.title('Hertzsprung-Russell Diagram by Stellar Class', fontsize=16)
    
    # Add grid
    plt.grid(alpha=0.3, linestyle='--')
    
    # Optimize legend for many classes
    if len(unique_classes) > 10:
        plt.legend(fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(1, 1))
    else:
        plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('figures/hr_diagram.png', dpi=300)
    #plt.show()


def plot_galactic_distribution(df, class_column, min_confidence=0.7, density_contours=True):
    """
    Create a plot showing the distribution of stars in Galactic coordinates.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    class_column : str
        The column name containing class labels
    min_confidence : float, optional
        Minimum confidence threshold to include in plot, default=0.7
    density_contours : bool, optional
        Whether to add density contours for each class, default=True
    """
    # Check if the necessary columns exist
    if 'l' not in df.columns or 'b' not in df.columns:
        raise ValueError("Dataframe must contain 'l' and 'b' columns for Galactic coordinates")
    
    plt.figure(figsize=(14, 8))
    
    df = df.copy()
    df[class_column] = df[class_column].astype(str)
    
    # Apply confidence threshold if the column exists
    confidence_col = class_column.replace('predicted_class', 'confidence')
    if confidence_col in df.columns:
        df = df[df[confidence_col] >= min_confidence]
    
    # Create a colormap for the different classes
    unique_classes = df[class_column].unique()
    cmap = plt.cm.get_cmap('tab20', len(unique_classes))
    
    # First, plot all points with low alpha to show overall distribution
    for i, class_name in enumerate(unique_classes):
        subset = df[df[class_column] == class_name]
        plt.scatter(subset['l'], subset['b'], 
                   label=class_name, alpha=0.3, s=5, c=[cmap(i)])
        
        # Add density contours if requested
        if density_contours and len(subset) > 100:
            try:
                # Calculate KDE for the class
                l_range = np.linspace(subset['l'].min(), subset['l'].max(), 100)
                b_range = np.linspace(subset['b'].min(), subset['b'].max(), 100)
                L, B = np.meshgrid(l_range, b_range)
                positions = np.vstack([L.ravel(), B.ravel()])
                
                from scipy.stats import gaussian_kde
                kernel = gaussian_kde(subset[['l', 'b']].values.T)
                Z = np.reshape(kernel(positions), L.shape)
                
                # Plot contours
                plt.contour(L, B, Z, colors=[cmap(i)], 
                           levels=np.linspace(Z.min(), Z.max(), 5)[1:], 
                           linewidths=2, alpha=0.7)
            except Exception as e:
                print(f"Could not create density contours for {class_name}: {e}")
    
    plt.xlabel('Galactic Longitude (l)', fontsize=14)
    plt.ylabel('Galactic Latitude (b)', fontsize=14)
    plt.title('Galactic Distribution of Stellar Classes', fontsize=16)
    
    # Add grid
    plt.grid(alpha=0.3, linestyle='--')
    
    # Determine the extent of the data for better axis limits
    l_range = df['l'].max() - df['l'].min()
    b_range = df['b'].max() - df['b'].min()
    plt.xlim(df['l'].min() - 0.05 * l_range, df['l'].max() + 0.05 * l_range)
    plt.ylim(df['b'].min() - 0.05 * b_range, df['b'].max() + 0.05 * b_range)
    
    # Optimize legend for many classes
    if len(unique_classes) > 10:
        plt.legend(fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(1, 1))
    else:
        plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('figures/galactic_distribution.png', dpi=300)
    #plt.show()
    
    # Create a density map as a 2D histogram for all stars
    plt.figure(figsize=(14, 8))
    
    # Calculate 2D histogram
    l_bins = np.linspace(df['l'].min(), df['l'].max(), 150)
    b_bins = np.linspace(df['b'].min(), df['b'].max(), 100)
    h, xedges, yedges = np.histogram2d(df['l'], df['b'], bins=[l_bins, b_bins])
    
    # Plot the 2D histogram
    plt.pcolormesh(xedges, yedges, h.T, cmap='viridis', norm=Normalize(vmin=0, vmax=np.percentile(h, 95)))
    
    cbar = plt.colorbar(pad=0.01)
    cbar.set_label('Star Count', rotation=270, labelpad=20)
    
    plt.xlabel('Galactic Longitude (l)', fontsize=14)
    plt.ylabel('Galactic Latitude (b)', fontsize=14)
    plt.title('Density Distribution of Stars in Galactic Coordinates', fontsize=16)
    plt.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('figures/galactic_density.png', dpi=300)
    #plt.show()


def plot_color_color(df, class_column, x_color=None, y_color=None, min_confidence=0.7):
    """
    Create a color-color diagram showing the relationship between two color indices.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    class_column : str
        The column name containing class labels
    x_color : str, optional
        The column name for the x-axis color index
    y_color : str, optional
        The column name for the y-axis color index
    min_confidence : float, optional
        Minimum confidence threshold to include in plot, default=0.7
    """
    # Determine appropriate columns if not specified
    color_candidates = [
        'j_med_mag-ks_med_mag',
        'h_med_mag-ks_med_mag',
        'z_med_mag-ks_med_mag',
        'y_med_mag-ks_med_mag'
    ]
    
    if x_color is None:
        for col in color_candidates:
            if col in df.columns:
                x_color = col
                break
        if x_color is None:
            raise ValueError("No suitable x-axis color index column found. Please specify x_color.")
    
    if y_color is None:
        for col in color_candidates:
            if col in df.columns and col != x_color:
                y_color = col
                break
        if y_color is None:
            raise ValueError("No suitable y-axis color index column found. Please specify y_color.")
    
    plt.figure(figsize=(12, 10))
    
    df = df.copy()
    df[class_column] = df[class_column].astype(str)
    
    # Apply confidence threshold if the column exists
    confidence_col = class_column.replace('predicted_class', 'confidence')
    if confidence_col in df.columns:
        df = df[df[confidence_col] >= min_confidence]
    
    # Remove outliers for better visualization
    q1_x = df[x_color].quantile(0.01)
    q3_x = df[x_color].quantile(0.99)
    q1_y = df[y_color].quantile(0.01)
    q3_y = df[y_color].quantile(0.99)
    
    df_filtered = df[(df[x_color] >= q1_x) & (df[x_color] <= q3_x) &
                    (df[y_color] >= q1_y) & (df[y_color] <= q3_y)]
    
    # Create a colormap for the different classes
    unique_classes = df_filtered[class_column].unique()
    cmap = plt.cm.get_cmap('tab20', len(unique_classes))
    
    # Create scatter plot
    for i, class_name in enumerate(unique_classes):
        subset = df_filtered[df_filtered[class_column] == class_name]
        plt.scatter(subset[x_color], subset[y_color], 
                   label=class_name, alpha=0.7, s=15, c=[cmap(i)])
    
    plt.xlabel(f'{x_color}', fontsize=14)
    plt.ylabel(f'{y_color}', fontsize=14)
    plt.title('Color-Color Diagram by Stellar Class', fontsize=16)
    
    # Add grid
    plt.grid(alpha=0.3, linestyle='--')
    
    # Optimize legend for many classes
    if len(unique_classes) > 10:
        plt.legend(fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(1, 1))
    else:
        plt.legend(fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    
    plt.tight_layout()
    plt.savefig('figures/color_color_diagram.png', dpi=300)
    #plt.show()


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
    plt.savefig('figures/feature_class_distributions.png', dpi=300)
    #plt.show()


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
    plt.savefig('figures/feature_importance.png', dpi=300)
    #plt.show()


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
    plt.savefig('figures/node_embeddings_tsne.png', dpi=300)
    
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
    plt.savefig('figures/node_embeddings_pca.png', dpi=300)


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
    plt.savefig('figures/confidence_distribution.png', dpi=300)
    #plt.show()


def plot_xgb_feature_importance(feature_names, importance_values, top_n=20):
    """
    Plot feature importance from XGBoost model.
    
    Parameters:
    -----------
    feature_names : list
        List of feature names
    importance_values : list
        List of importance values
    top_n : int, optional
        Number of top features to show, default=20
    """
    # Sort features by importance
    indices = np.argsort(importance_values)[-top_n:]
    top_features = [feature_names[i] for i in indices]
    top_importance = [importance_values[i] for i in indices]
    
    plt.figure(figsize=(14, 10))
    plt.barh(range(len(top_features)), top_importance, align='center', color='skyblue', edgecolor='navy', alpha=0.8)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance (Gain)', fontsize=14)
    plt.title(f'Top {top_n} Feature Importance', fontsize=16)
    plt.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('figures/xgb_feature_importance.png', dpi=300)
    #plt.show()


def plot_xgb_class_probability_heatmap(probs, class_names, max_samples=100):
    """
    Create a heatmap of class probabilities for a subset of samples.
    
    Parameters:
    -----------
    probs : numpy.ndarray
        Probability matrix from the classifier (n_samples, n_classes)
    class_names : list
        List of class names
    max_samples : int, optional
        Maximum number of samples to include in heatmap, default=100
    """
    # Sample a subset of probability predictions for visualization
    n_samples = min(max_samples, probs.shape[0])
    indices = np.random.choice(probs.shape[0], size=n_samples, replace=False)
    sampled_probs = probs[indices]
    
    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    im = ax.imshow(sampled_probs, aspect='auto', cmap='viridis')
    
    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2%", pad=0.1)
    plt.colorbar(im, cax=cax)
    
    # Set labels
    ax.set_yticks(range(n_samples))
    ax.set_yticklabels([f"Sample {i}" for i in range(n_samples)])
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    
    plt.title('Class Probability Heatmap (Sample of Predictions)', fontsize=16)
    plt.tight_layout()
    plt.savefig('figures/xgb_class_probabilities.png', dpi=300)
    #plt.show()


def plot_xgb_top2_confidence_scatter(probs, preds, class_names):
    """
    Create a scatter plot showing the top 2 class probabilities for each prediction.
    
    Parameters:
    -----------
    probs : numpy.ndarray
        Probability matrix from the classifier (n_samples, n_classes)
    preds : numpy.ndarray
        Predicted class indices
    class_names : list
        List of class names
    """
    # Get top 2 probabilities for each prediction
    sorted_probs = np.sort(probs, axis=1)
    top1_probs = sorted_probs[:, -1]
    top2_probs = sorted_probs[:, -2]
    confidence_gap = top1_probs - top2_probs
    
    plt.figure(figsize=(12, 10))
    
    # Create scatter plot with color representing the predicted class
    scatter = plt.scatter(top1_probs, top2_probs, c=preds, cmap='tab20', alpha=0.7, s=20)
    
    # Add colorbar legend
    legend1 = plt.colorbar(scatter)
    legend1.set_label('Predicted Class')
    
    # Add diagonal reference line
    max_val = max(np.max(top1_probs), np.max(top2_probs))
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    plt.xlabel('Top Class Probability', fontsize=14)
    plt.ylabel('Second Class Probability', fontsize=14)
    plt.title('Decision Confidence Analysis', fontsize=16)
    plt.grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('figures/xgb_confidence_analysis.png', dpi=300)
    
    # Create a second plot showing the confidence gap distribution by class
    plt.figure(figsize=(12, 6))
    for i, class_name in enumerate(class_names):
        class_gaps = confidence_gap[preds == i]
        if len(class_gaps) > 0:
            sns.kdeplot(class_gaps, label=f"{class_name}", alpha=0.7)
    
    plt.xlabel('Confidence Gap (Top - Second Probability)', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title('Confidence Gap Distribution by Predicted Class', fontsize=16)
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('figures/xgb_confidence_gap.png', dpi=300)
    #plt.show()


def plot_astronomical_map(df, class_column, min_confidence=0.7):
    """
    Create a map of astronomical objects in Galactic coordinates, color-coded by class.
    This is an alias for plot_galactic_distribution for compatibility.
    """
    plot_galactic_distribution(df, class_column, min_confidence)


def plot_misclassification_analysis(y_true, y_pred, probs, class_names):
    """
    Analyze misclassified samples by examining their class probabilities.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    probs : numpy.ndarray
        Probability matrix from the classifier (n_samples, n_classes)
    class_names : list
        List of class names
    """
    # Convert class labels to numeric if they're not already
    if not isinstance(y_true[0], (int, np.integer)):
        label_encoder = LabelEncoder()
        y_true_encoded = label_encoder.fit_transform(y_true)
        y_pred_encoded = label_encoder.transform(y_pred)
    else:
        y_true_encoded = y_true
        y_pred_encoded = y_pred
    
    # Identify misclassified samples
    misclassified = y_true_encoded != y_pred_encoded
    
    # If there are no misclassifications, return
    if not np.any(misclassified):
        print("No misclassifications found to analyze")
        return
    
    # Get probabilities for true and predicted classes
    true_class_probs = np.zeros(len(y_true_encoded))
    pred_class_probs = np.zeros(len(y_pred_encoded))
    
    for i in range(len(y_true_encoded)):
        true_class_probs[i] = probs[i, y_true_encoded[i]]
        pred_class_probs[i] = probs[i, y_pred_encoded[i]]
    
    # Create scatter plot of true vs predicted probabilities for misclassified samples
    plt.figure(figsize=(12, 10))
    
    plt.scatter(true_class_probs[misclassified], pred_class_probs[misclassified], 
               c=y_pred_encoded[misclassified], cmap='tab20', alpha=0.7, s=30)
    
    plt.axline([0, 0], [1, 1], linestyle='--', color='r', alpha=0.5)
    
    plt.xlabel('True Class Probability', fontsize=14)
    plt.ylabel('Predicted Class Probability', fontsize=14)
    plt.title('Analysis of Misclassified Samples', fontsize=16)
    plt.grid(alpha=0.3, linestyle='--')
    
    plt.colorbar(label='Predicted Class')
    
    plt.tight_layout()
    plt.savefig('figures/misclassification_analysis.png', dpi=300)
    
    # Create a histogram of misclassifications by true class
    plt.figure(figsize=(14, 8))
    
    # Calculate misclassification counts by true class
    true_classes, true_counts = np.unique(y_true_encoded[misclassified], return_counts=True)
    
    # Sort by counts for better visualization
    sorted_indices = np.argsort(true_counts)[::-1]
    sorted_classes = true_classes[sorted_indices]
    sorted_counts = true_counts[sorted_indices]
    
    # Create bar chart
    bars = plt.bar(
        [class_names[cls] for cls in sorted_classes], 
        sorted_counts, 
        color='skyblue', 
        edgecolor='navy', 
        alpha=0.8
    )
    
    plt.xlabel('True Class', fontsize=14)
    plt.ylabel('Number of Misclassifications', fontsize=14)
    plt.title('Misclassification Counts by True Class', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height + 0.1,
            f'{height:.0f}',
            ha='center', va='bottom'
        )
    
    plt.tight_layout()
    plt.savefig('figures/misclassification_by_class.png', dpi=300)
    
    # Create confusion matrix specifically for misclassified samples
    cm_misc = confusion_matrix(y_true_encoded[misclassified], y_pred_encoded[misclassified])
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_misc, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix for Misclassified Samples Only')
    plt.tight_layout()
    plt.savefig('figures/misclassification_confusion_matrix.png', dpi=300)
    #plt.show()

# New functions from paste.txt, adapted for GXGB.py

def plot_galactic_coords(df, class_column, output_dir='class_figures', min_prob=0.7, min_confidence=0.9, max_entropy=0.2):
    """
    Create a scatter plot of stars in galactic coordinates colored by class type.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    class_column : str
        The column name containing class labels
    output_dir : str, optional
        Directory to save output figures, default='class_figures'
    min_prob : float, optional
        Minimum probability threshold, default=0.7
    min_confidence : float, optional
        Minimum confidence metric threshold, default=0.9
    max_entropy : float, optional
        Maximum entropy threshold, default=0.2
    """
    # Set up figure parameters
    set_plot_style(large_text=True)
    plt.rcParams['legend.fontsize'] = 15  # Specific legend size
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Apply filters based on confidence thresholds
    prob_col = class_column.replace('predicted_class', 'confidence')
    if prob_col in df.columns:
        df = df[df[prob_col] > min_prob]
    
    # Additional confidence columns if they exist
    if 'variable_confidence_metric' in df.columns:
        df = df[df['variable_confidence_metric'] > min_confidence]
    if 'variable_entropy' in df.columns:
        df = df[df['variable_entropy'] < max_entropy]
        
    # Sample data to avoid overcrowding
    sampled_df = df.groupby(class_column).apply(
        lambda x: x.nlargest(n=min(len(x), 100000), columns=prob_col)
    ).reset_index(drop=True)
    
    # Create categorical encoding for class types
    sampled_df['type_code'] = pd.Categorical(sampled_df[class_column]).codes
    
    # Set up colors and markers for different classes
    unique_types = sampled_df[class_column].unique()
    colors = ['red', 'k', 'darkgreen', 'yellowgreen', 'plum', 'goldenrod', 'fuchsia', 'navy', 'turquoise', 'aqua', 'k']
    markers = ['o', '^', 'X', '*', 's', 'P', '^', 'D', '<', '>']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 9))
    legend_handles = []
    
    for i, var_type in enumerate(unique_types):
        type_df = sampled_df[sampled_df[class_column] == var_type]
        
        # Customize appearance based on class
        alpha = 0.7
        s = 2
        if i in [0, 3, 4, 5]:
            alpha = 1
        if i in [0, 3, 5]:
            s = 5
            
        # Plot the class
        plt.scatter(type_df['l'], type_df['b'], 
                   color=colors[i % len(colors)], 
                   marker=markers[i % len(markers)], 
                   label=var_type, s=s, alpha=alpha)
        
        # Add to legend
        legend_handles.append(plt.Line2D([0], [0], marker=markers[i % len(markers)], 
                                       color='w', markerfacecolor=colors[i % len(colors)], 
                                       markersize=s*4, label=var_type))
    
    # Customize axes
    ax.xaxis.tick_top()  # Move x-axis to the top
    ax.xaxis.set_label_position('top')  # Move x-axis label to the top
    ax.set_xlabel('Galactic longitude (deg)')
    ax.set_ylabel('Galactic latitude (deg)')
    
    # Set axis limits
    ax.set_xlim(min(sampled_df['l'].values), max(sampled_df['l'].values))
    ax.set_ylim(min(sampled_df['b'].values), max(sampled_df['b'].values))
    ax.invert_xaxis()
    
    # Add legend
    plt.legend(ncol=2, handles=legend_handles)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/galactic_coordinates.jpg', dpi=300, bbox_inches='tight')
    plt.close()


def plot_bailey_diagram(df, class_column, output_dir='class_figures', min_prob=0.7, min_confidence=0.9, max_entropy=0.2):
    """
    Create a Bailey diagram showing period vs amplitude for variable stars, colored by class.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    class_column : str
        The column name containing class labels
    output_dir : str, optional
        Directory to save output figures, default='class_figures'
    min_prob : float, optional
        Minimum probability threshold, default=0.7
    min_confidence : float, optional
        Minimum confidence metric threshold, default=0.9
    max_entropy : float, optional
        Maximum entropy threshold, default=0.2
    """
    # Set up figure parameters
    set_plot_style(large_text=True)
    plt.rcParams['legend.fontsize'] = 15  # Specific legend size
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Apply filters
    df = df[df['true_amplitude'] < 2]  # Filter extreme amplitudes
    
    # Add log period if not present
    if 'log_true_period' not in df.columns:
        df['log_true_period'] = np.log10(df['true_period'])
    
    df = df[df['log_true_period'] < 2.7]  # Filter period range
    
    # Apply confidence thresholds
    prob_col = class_column.replace('predicted_class', 'confidence')
    if prob_col in df.columns:
        df = df[df[prob_col] > min_prob]
    
    # Additional confidence columns if they exist
    if 'variable_confidence_metric' in df.columns:
        df = df[df['variable_confidence_metric'] > min_confidence]
    if 'variable_entropy' in df.columns:
        df = df[df['variable_entropy'] < max_entropy]
    
    # Create bin edges for potential density plot
    lon_edges = np.linspace(-1, 2.7, 100)
    mag_edges = np.linspace(0, 2, 100)
    
    # Sample to avoid overcrowding
    sampled_df = df.groupby(class_column).apply(
        lambda x: x.nlargest(n=min(len(x), 10000), columns=prob_col)
    ).reset_index(drop=True)
    
    # Create categorical encoding
    sampled_df['type_code'] = pd.Categorical(sampled_df[class_column]).codes
    
    # Set up colors and markers
    unique_types = sampled_df[class_column].unique()
    colors = ['red', 'k', 'darkgreen', 'yellowgreen', 'plum', 'goldenrod', 'fuchsia', 'navy', 'turquoise', 'aqua', 'k']
    markers = ['o', '^', 'X', '*', 's', 'P', '^', 'D', '<', '>']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    for i, var_type in enumerate(unique_types):
        type_df = sampled_df[sampled_df[class_column] == var_type]
        
        # Customize appearance based on class
        alpha = 0.7
        s = 40
        if i in [3, 4, 5]:
            alpha = 1
        if i == 3:
            s = 100
            
        # Plot the data
        plt.scatter(type_df['log_true_period'], type_df['true_amplitude'], 
                   color=colors[i % len(colors)], 
                   marker=markers[i % len(markers)], 
                   label=var_type, s=s, alpha=alpha)
    
    # Add labels and legend
    plt.legend(bbox_to_anchor=(0.1, 0.8), ncol=2)
    ax.set_xlabel(r'log$_{10}$(Period) [days]')
    ax.set_ylabel(r'Amplitude [mag]')
    
    # Set axis limits
    ax.set_xlim(-1, 2.7)
    ax.set_ylim(0, 2)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig('figures/bailey_diagram.jpg', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_entropy(df, class_column, output_dir='class_figures', min_prob=0.7):
    """
    Create a scatter plot and heatmap showing relationship between confidence metric and entropy.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The dataframe containing the data
    class_column : str
        The column name containing class labels
    output_dir : str, optional
        Directory to save output figures, default='class_figures'
    min_prob : float, optional
        Minimum probability threshold, default=0.7
    """
    # Set up figure parameters
    set_plot_style(large_text=True) 
    plt.rcParams['legend.fontsize'] = 19
    plt.rcParams['axes.labelsize'] = 25
    plt.rcParams['lines.markersize'] = 15
    
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Apply filters based on probability
    prob_col = class_column.replace('predicted_class', 'confidence')
    if prob_col in df.columns:
        df = df[df[prob_col] > min_prob]
    
    # Set up bin edges for visualization
    lon_edges = np.linspace(0.4, 1, 100)
    mag_edges = np.linspace(0, 1.3, 100)
    
    # Create figure with two subplots (vertical stacking)
    fig = plt.figure(figsize=(15, 20))
    
    # Create scatter plot (top)
    ax1 = fig.add_axes([0, 0.5, 1, 0.5])
    
    # Sample data to avoid overcrowding
    sampled_df = df.groupby(class_column).apply(
        lambda x: x.nlargest(n=min(len(x), 1000), columns=prob_col)
    ).reset_index(drop=True)
    
    # Set up colors and markers
    unique_types = sampled_df[class_column].unique()
    colors = ['red', 'k', 'darkgreen', 'yellowgreen', 'plum', 'grey', 'goldenrod', 
             'brown', 'fuchsia', 'navy', 'turquoise', 'aqua', 'k']
    markers = ['o', '^', 'X', '*', 's', 'p', 'P', 'v', '^', 'D', '<', '>']
    
    # Check if the required confidence columns exist
    if 'variable_confidence_metric' not in df.columns or 'variable_entropy' not in df.columns:
        # Adapt to use available columns
        x_col = prob_col  # Use probability as x
        if len(df.columns) > 1:
            # Try to find another numeric column for y
            for col in df.columns:
                if col != prob_col and pd.api.types.is_numeric_dtype(df[col]) and col != class_column:
                    y_col = col
                    break
            else:
                # If no suitable y column, create a random one for demonstration
                df['uncertainty'] = 1 - df[prob_col]
                y_col = 'uncertainty'
        else:
            # Create random entropy-like values
            df['uncertainty'] = 1 - df[prob_col] + np.random.normal(0, 0.1, size=len(df))
            df['uncertainty'] = np.clip(df['uncertainty'], 0, 1.3)
            y_col = 'uncertainty'
    else:
        # Use the actual confidence columns
        x_col = 'variable_confidence_metric'
        y_col = 'variable_entropy'
    
    # Plot points
    for i, var_type in enumerate(unique_types):
        type_df = sampled_df[sampled_df[class_column] == var_type]
        
        # Customize appearance
        alpha = 0.7
        s = 80
        if i in [3, 4, 5]:
            alpha = 1
        if i == 3:
            s = 150
            
        # Plot data
        ax1.scatter(
            type_df[x_col] if x_col in type_df.columns else type_df[prob_col],
            type_df[y_col] if y_col in type_df.columns else 1-type_df[prob_col],
            color=colors[i % len(colors)], 
            marker=markers[i % len(markers)], 
            label=var_type, s=s, alpha=alpha
        )
    
    # Add legend and labels
    ax1.legend(ncol=2, loc='lower left')
    ax1.set_ylabel(r'Entropy' if y_col == 'variable_entropy' else y_col)
    
    # Set axis limits
    if x_col == 'variable_confidence_metric':
        ax1.set_xlim([lon_edges[0], lon_edges[-1]])
    else:
        ax1.set_xlim([0.4, 1])
        
    # Remove x-axis labels from top plot
    ax1.tick_params(labelbottom=False)
    
    # Create heatmap (bottom) 
    ax2 = fig.add_axes([0, 0, 1, 0.5])
    
    # Create 2D histogram
    H, xedges, yedges = np.histogram2d(
        df[x_col] if x_col in df.columns else df[prob_col],
        df[y_col] if y_col in df.columns else 1-df[prob_col],
        bins=[lon_edges, mag_edges]
    )
    
    # Plot heatmap
    ax2.imshow(H.T, origin='lower', 
              extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
              aspect='auto', cmap='inferno', norm=LogNorm())
    
    # Add labels
    ax2.set_xlabel(r'Confidence metric' if x_col == 'variable_confidence_metric' else x_col)
    ax2.set_ylabel(r'Entropy' if y_col == 'variable_entropy' else y_col)
    
    # Set axis limits for bottom plot
    if x_col == 'variable_confidence_metric':
        ax2.set_xlim([lon_edges[0], lon_edges[-1]])
    else:
        ax2.set_xlim([0.4, 1])
        
    # Ensure bottom plot has x-axis labels
    ax2.tick_params(top=False, labeltop=False)
    
    # Save figure
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/classification_confidence.jpg', dpi=300, bbox_inches='tight')
    plt.close()






def visualize_pca(pca_df, original_df, pca_model, output_dir='./figures'):
    # Plot standard PCA views
    os.makedirs(output_dir, exist_ok=True)

    # Scatter
    plt.figure(); plt.scatter(pca_df['PC1'], pca_df['PC2'], s=5, alpha=0.5)
    plt.xlabel(f"PC1 ({pca_model.explained_variance_ratio_[0]:.2%})")
    plt.ylabel(f"PC2 ({pca_model.explained_variance_ratio_[1]:.2%})")
    plt.title("PCA: PC1 vs PC2")
    plt.grid(); plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_scatter.png"); plt.close()

    # Density
    plt.figure(); sns.kdeplot(x=pca_df['PC1'], y=pca_df['PC2'], cmap="Blues", fill=True)
    plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("PCA Density"); plt.grid()
    plt.tight_layout(); plt.savefig(f"{output_dir}/pca_density.png"); plt.close()

    # 2D Hist
    plt.figure(); h = plt.hist2d(pca_df['PC1'], pca_df['PC2'], bins=100, cmap='viridis', norm=LogNorm())
    plt.colorbar(h[3]); plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("PCA Histogram"); plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_histogram.png"); plt.close()

    # Scree
    plt.figure()
    plt.bar(range(1, len(pca_model.explained_variance_ratio_)+1), pca_model.explained_variance_ratio_)
    plt.step(range(1, len(pca_model.explained_variance_ratio_)+1), np.cumsum(pca_model.explained_variance_ratio_), label='Cumulative')
    plt.xlabel('PC'); plt.ylabel('Explained Variance'); plt.legend()
    plt.tight_layout(); plt.savefig(f"{output_dir}/pca_scree.png"); plt.close()

    # Loadings
    load = pd.DataFrame(pca_model.components_.T, index=pca_model.feature_names_in_, columns=[f'PC{i+1}' for i in range(pca_model.n_components_)])
    top = pd.concat([load['PC1'].abs().sort_values(ascending=False), load['PC2'].abs().sort_values(ascending=False)]).index.unique()[:15]
    plt.figure(figsize=(10, 6)); sns.heatmap(load.loc[top, ['PC1', 'PC2']], annot=True, cmap='coolwarm', center=0)
    plt.title("Top Feature Loadings"); plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_loadings.png"); plt.close()

    # Coloured scatter: FAP, period, amplitude
    if 'best_fap' in original_df.columns:
        plt.figure()
        plt.scatter(pca_df['PC1'], pca_df['PC2'], c=original_df['best_fap'], cmap='viridis_r', s=10, alpha=0.7)
        plt.colorbar(label='FAP'); plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_fap.png"); plt.close()

    if 'true_period' in original_df.columns:
        plt.figure()
        plt.scatter(pca_df['PC1'], pca_df['PC2'], c=np.log10(np.clip(original_df['true_period'].values, 0.01, None)), cmap='plasma', s=10, alpha=0.7)
        plt.colorbar(label='log10(Period)'); plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_period.png"); plt.close()

    if 'true_amplitude' in original_df.columns:
        plt.figure()
        plt.scatter(pca_df['PC1'], pca_df['PC2'], c=np.log10(np.clip(original_df['true_amplitude'].values, 0.001, None)), cmap='inferno', s=10, alpha=0.7)
        plt.colorbar(label='log10(Amplitude)'); plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_amplitude.png"); plt.close()

