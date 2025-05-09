import networkx as nx
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

