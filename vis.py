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


def plot_xgb_feature_importance(feature_names, importance_scores, top_n=20):
    """
    Create a better visualization of XGBoost feature importance.
    
    Parameters:
    -----------
    feature_names : list
        List of feature names
    importance_scores : list
        List of importance scores corresponding to the features
    top_n : int
        Number of top features to display
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Sort by importance
    indices = np.argsort(importance_scores)[::-1]
    
    # Select top N features
    top_indices = indices[:top_n]
    top_features = [feature_names[i] for i in top_indices]
    top_scores = [importance_scores[i] for i in top_indices]
    
    # Create horizontal bar plot
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(top_scores)), top_scores, align='center', color='skyblue', edgecolor='navy', alpha=0.8)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance Score (gain)')
    plt.title('XGBoost Feature Importance', fontsize=16)
    plt.grid(axis='x', alpha=0.3)
    
    # Add values to the end of each bar
    for i, v in enumerate(top_scores):
        plt.text(v + v*0.01, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    plt.savefig('xgb_feature_importance.png', dpi=300)
    plt.close()

def plot_xgb_class_probability_heatmap(probs, class_names, num_samples=50):
    """
    Plot a heatmap of class probabilities for a subset of test samples.
    
    Parameters:
    -----------
    probs : numpy.ndarray
        Probability matrix with shape (n_samples, n_classes)
    class_names : list
        List of class names
    num_samples : int
        Number of random samples to show in the heatmap
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Select a random subset of samples
    if probs.shape[0] > num_samples:
        indices = np.random.choice(probs.shape[0], num_samples, replace=False)
        sample_probs = probs[indices]
    else:
        sample_probs = probs
        num_samples = probs.shape[0]
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(sample_probs, cmap='viridis', 
                    yticklabels=range(1, num_samples+1),
                    xticklabels=class_names)
    plt.xlabel('Classes')
    plt.ylabel('Samples')
    plt.title('Class Probabilities Heatmap', fontsize=16)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('xgb_class_probabilities.png', dpi=300)
    plt.close()

def plot_xgb_top2_confidence_scatter(probs, preds, class_names):
    """
    Create a scatter plot showing the confidence of top 1 vs top 2 predictions.
    This helps visualize the model's confidence in its predictions.
    
    Parameters:
    -----------
    probs : numpy.ndarray
        Probability matrix with shape (n_samples, n_classes)
    preds : numpy.ndarray
        Predicted class indices
    class_names : list
        List of class names
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Sort probabilities to get top 2 values for each sample
    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    top1_confidence = sorted_probs[:, 0]  # Highest probability
    top2_confidence = sorted_probs[:, 1]  # Second highest probability
    confidence_gap = top1_confidence - top2_confidence  # Gap between top 2 confidences
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Color points by predicted class
    for i, class_name in enumerate(class_names):
        mask = preds == i
        if np.any(mask):  # Check if we have any predictions for this class
            plt.scatter(top1_confidence[mask], 
                       confidence_gap[mask], 
                       alpha=0.6, s=50, label=class_name)
    
    plt.xlabel('Top Class Confidence')
    plt.ylabel('Confidence Gap (Top1 - Top2)')
    plt.title('Model Decision Confidence Analysis', fontsize=16)
    plt.grid(alpha=0.3)
    
    # Add legend with reasonable size based on number of classes
    num_classes = len(class_names)
    if num_classes <= 10:
        plt.legend(fontsize=10)
    elif num_classes <= 20:
        plt.legend(fontsize=8, ncol=2)
    else:
        plt.legend(fontsize=6, ncol=3)
    
    plt.tight_layout()
    plt.savefig('xgb_confidence_analysis.png', dpi=300)
    plt.close()

def plot_astronomical_map(df, class_column, l_column='l', b_column='b', sample_size=10000):
    """
    Create a plot of objects in Galactic coordinates, colored by class.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data
    class_column : str
        Column name for the class labels
    l_column : str
        Column name for Galactic longitude
    b_column : str
        Column name for Galactic latitude
    sample_size : int
        Number of points to sample (for large datasets)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # Sample data if needed
    if len(df) > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
    else:
        df_sample = df
    
    # Create figure
    plt.figure(figsize=(14, 10))
    
    # Get unique classes for coloring
    unique_classes = df_sample[class_column].unique()
    
    # Create scatter plot grouped by class
    for i, class_name in enumerate(unique_classes):
        subset = df_sample[df_sample[class_column] == class_name]
        plt.scatter(subset[l_column], subset[b_column], 
                   s=10, alpha=0.7, label=class_name)
    
    plt.xlabel('Galactic Longitude (l)')
    plt.ylabel('Galactic Latitude (b)')
    plt.title('Spatial Distribution of Classes in the Galaxy', fontsize=16)
    
    # Add horizontal line at b=0 to indicate Galactic plane
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    
    # Customize legend based on number of classes
    num_classes = len(unique_classes)
    if num_classes <= 8:
        plt.legend(fontsize=10)
    elif num_classes <= 15:
        plt.legend(fontsize=8, ncol=2)
    else:
        plt.legend(fontsize=6, ncol=3)
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('galactic_distribution.png', dpi=300)
    plt.close()

def plot_misclassification_analysis(y_true, y_pred, probs, class_names):
    """
    Analyze and visualize patterns in misclassifications.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    probs : numpy.ndarray
        Probability matrix with shape (n_samples, n_classes)
    class_names : list
        List of class names
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    
    # Identify misclassified samples
    misclassified = y_true != y_pred
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_true, y_pred, normalize='true')
    
    # Create mask for diagonal (correctly classified) to highlight misclassifications
    mask = np.eye(len(class_names), dtype=bool)
    
    # Plot the confusion matrix with a diverging colormap
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='coolwarm', 
               xticklabels=class_names, yticklabels=class_names,
               mask=mask, cbar_kws={'label': 'Fraction of samples'})
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Misclassification Analysis (Off-diagonal Elements)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('misclassification_analysis.png', dpi=300)
    plt.close()
    
    # Get confidence of misclassified samples
    if np.any(misclassified):
        conf_misc = np.max(probs[misclassified], axis=1)
        
        # Plot histogram of confidences for misclassified samples
        plt.figure(figsize=(10, 8))
        plt.hist(conf_misc, bins=20, alpha=0.7, color='red', 
                edgecolor='black', label='Misclassified')
        
        # Add histogram of correctly classified samples for comparison
        conf_correct = np.max(probs[~misclassified], axis=1)
        plt.hist(conf_correct, bins=20, alpha=0.5, color='blue', 
                edgecolor='black', label='Correctly classified')
        
        plt.xlabel('Model Confidence')
        plt.ylabel('Number of Samples')
        plt.title('Confidence Distribution: Misclassified vs. Correctly Classified', fontsize=16)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('misclassification_confidence.png', dpi=300)
        plt.close()


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



def plot_period_amplitude(df, class_column):
    plt.figure(figsize=(12, 10))
    
    df = df.copy()
    df[class_column] = df[class_column].astype(str)  # Convert to string to avoid endianness issues
    
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

