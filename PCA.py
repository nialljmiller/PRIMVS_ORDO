#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import seaborn as sns
from astropy.io import fits
import argparse
import os
from matplotlib.colors import LogNorm

def load_data(filepath):
    """
    Load data from a FITS file or CSV file
    """
    print(f"Loading data from {filepath}...")
    
    if filepath.endswith('.fits'):
        # Load FITS file
        with fits.open(filepath) as hdul:
            data = hdul[1].data
            try:
                df = pd.DataFrame(data)
            except Exception as e:
                print(f"Direct conversion failed: {e}")
                try:
                    df = pd.DataFrame.from_records(data)
                except Exception as e:
                    print(f"Structured array approach failed: {e}")
                    columns = data.names
                    df = pd.DataFrame()
                    for col in columns:
                        df[col] = data[col]
    else:
        # Assume CSV format
        df = pd.DataFrame(pd.read_csv(filepath))
    
    print(f"Loaded {len(df)} records with {len(df.columns)} features")
    return df

def identify_feature_columns(df):
    """
    Identify different feature categories from the dataframe
    """
    # Physical features from Table 4 in the paper
    PHYSICAL_FEATURES = [
        # Color features
        "z_med_mag-ks_med_mag", "y_med_mag-ks_med_mag", 
        "j_med_mag-ks_med_mag", "h_med_mag-ks_med_mag",
        # Positional features
        "l", "b", 
        # Statistical features
        "Cody_M", "stet_k", "eta_e", "med_BRP", "range_cum_sum",
        "max_slope", "MAD", "mean_var", "percent_amp", "true_amplitude",
        "roms", "p_to_p_var", "lag_auto", "AD", "std_nxs",
        # Weighted statistics
        "weight_mean", "weight_std", "weight_skew", "weight_kurt",
        # Basic statistics
        "mean", "std", "skew", "kurt", 
        # Period
        "true_period"
    ]
    
    # Check which physical features are actually present in the data
    available_physical = [f for f in PHYSICAL_FEATURES if f in df.columns]
    
    # Look for embedding features (numbered dimensions from contrastive learning)
    embedding_features = [str(i) for i in range(128) if str(i) in df.columns]
    
    # Additional features that might be useful
    other_features = []
    for feature in ["fap", "best_fap", "ls_fap", "pdm_fap", "ce_fap", "gp_fap"]:
        if feature in df.columns:
            other_features.append(feature)
    
    # Identify classification or target columns if they exist
    target_columns = []
    for col in df.columns:
        if "class" in col.lower() or "label" in col.lower() or "type" in col.lower():
            target_columns.append(col)
    
    # Print summary
    print(f"Found {len(available_physical)} physical features")
    print(f"Found {len(embedding_features)} embedding features")
    print(f"Found {len(other_features)} additional features")
    if target_columns:
        print(f"Found potential target columns: {', '.join(target_columns)}")
    
    return {
        'physical': available_physical,
        'embeddings': embedding_features,
        'additional': other_features,
        'targets': target_columns
    }

def prepare_data(df, feature_groups, scale_method='robust', impute_method='knn'):
    """
    Prepare data for PCA:
    1. Select features to use
    2. Handle missing values
    3. Handle outliers 
    4. Scale features
    5. Optional: Transform features for better normality
    """
    # Combine all features except targets
    all_features = feature_groups['physical'] + feature_groups['embeddings'] + feature_groups['additional']
    
    # Create a copy of the dataframe with only the needed columns
    df_selected = df[all_features].copy()
    
    # Handle missing values
    print(f"Missing values before imputation: {df_selected.isna().sum().sum()}")
    
    # Replace infinities with NaN
    df_selected.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if impute_method == 'knn':
        # KNN imputation (better but slower)
        imputer = KNNImputer(n_neighbors=5)
        df_selected_imputed = pd.DataFrame(
            imputer.fit_transform(df_selected),
            columns=df_selected.columns
        )
    else:
        # Simple median imputation
        df_selected_imputed = df_selected.fillna(df_selected.median())
    
    print(f"Missing values after imputation: {df_selected_imputed.isna().sum().sum()}")
    
    # Scale features
    if scale_method == 'robust':
        # Robust scaling (less sensitive to outliers)
        scaler = RobustScaler()
    elif scale_method == 'standard':
        # Standard scaling (z-score normalization)
        scaler = StandardScaler()
    else:
        raise ValueError("Unknown scaling method")
    
    # Apply scaling
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_selected_imputed),
        columns=df_selected_imputed.columns
    )
    
    # Optional: Apply transformation for better normality
    if 'true_period' in df_scaled.columns:
        # Log transform period (commonly done for astronomical periods)
        period_col = df_scaled.columns.get_loc('true_period')
        # Add a small constant to avoid log(0)
        min_positive = df_selected_imputed['true_period'][df_selected_imputed['true_period'] > 0].min()
        period_data = df_selected_imputed['true_period'].values
        period_data[period_data <= 0] = min_positive / 10
        df_scaled.iloc[:, period_col] = scaler.fit_transform(
            np.log10(period_data).reshape(-1, 1)
        )
    
    return df_scaled, df_selected_imputed

def apply_pca(data, n_components=2, random_state=42):
    """
    Apply PCA to reduce dimensionality
    """
    # Initialize PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    
    # Fit and transform data
    principal_components = pca.fit_transform(data)
    
    # Create dataframe with principal components
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(n_components)]
    )
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    print(f"Explained variance by component: {explained_variance}")
    print(f"Cumulative explained variance: {cumulative_variance}")
    
    return pca_df, pca

def visualize_pca(pca_df, original_df, feature_groups, pca_model, output_dir='./figures'):
    """
    Create various visualizations of the PCA results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Simple scatter plot of PC1 vs PC2
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_df['PC1'], pca_df['PC2'], alpha=0.5, s=5)
    plt.title('PCA: First Two Principal Components')
    plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_scatter.png", dpi=300)
    plt.close()
    
    # Plot 2: Density plot
    plt.figure(figsize=(10, 8))
    sns.kdeplot(x=pca_df['PC1'], y=pca_df['PC2'], cmap="Blues", fill=True)
    plt.title('Density Distribution of First Two Principal Components')
    plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_density.png", dpi=300)
    plt.close()
    
    # Plot 3: 2D histogram (better for large datasets)
    plt.figure(figsize=(10, 8))
    h = plt.hist2d(pca_df['PC1'], pca_df['PC2'], bins=100, cmap='viridis', norm=LogNorm())
    plt.colorbar(h[3], label='Count (log scale)')
    plt.title('2D Histogram of First Two Principal Components')
    plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_histogram.png", dpi=300)
    plt.close()
    
    # Plot 4: Explained variance ratio
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(pca_model.explained_variance_ratio_) + 1), 
            pca_model.explained_variance_ratio_, alpha=0.7)
    plt.step(range(1, len(pca_model.explained_variance_ratio_) + 1), 
            np.cumsum(pca_model.explained_variance_ratio_), where='mid', 
            label='Cumulative Explained Variance')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot - Explained Variance by Principal Component')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_scree_plot.png", dpi=300)
    plt.close()
    
    # Plot 5: Feature contributions (loadings) to first 2 PCs
    plt.figure(figsize=(12, 8))
    loadings = pd.DataFrame(
        pca_model.components_.T,
        columns=[f'PC{i+1}' for i in range(pca_model.components_.shape[0])],
        index=pca_df.columns
    )
    
    # Only show top 20 features by absolute contribution to PC1 or PC2
    top_features = pd.concat([
        loadings['PC1'].abs().sort_values(ascending=False).head(10),
        loadings['PC2'].abs().sort_values(ascending=False).head(10)
    ]).index.unique()
    
    sns.heatmap(loadings.loc[top_features, ['PC1', 'PC2']], 
                annot=True, cmap='coolwarm', fmt=".2f", center=0)
    plt.title('Feature Contributions to PC1 and PC2')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_feature_contributions.png", dpi=300)
    plt.close()
    
    # If there are target columns (classes), create colored plots
    for target_col in feature_groups['targets']:
        if target_col in original_df.columns:
            plt.figure(figsize=(12, 10))
            
            # Check the target type
            if original_df[target_col].dtype == 'O' or len(original_df[target_col].unique()) < 10:
                # Categorical target
                target_unique = original_df[target_col].unique()
                
                # Limit to 10 classes for clarity if there are too many
                if len(target_unique) > 10:
                    # Get the 9 most common classes and group others
                    top_classes = original_df[target_col].value_counts().nlargest(9).index
                    original_df[f'{target_col}_grouped'] = original_df[target_col].apply(
                        lambda x: x if x in top_classes else 'Other'
                    )
                    target_col = f'{target_col}_grouped'
                
                # Assign colors and plot each class
                target_unique = original_df[target_col].unique()
                for i, target_value in enumerate(target_unique):
                    subset = original_df[original_df[target_col] == target_value]
                    idx = subset.index
                    plt.scatter(
                        pca_df.loc[idx, 'PC1'], 
                        pca_df.loc[idx, 'PC2'],
                        alpha=0.7, s=10, label=target_value
                    )
                plt.legend(title=target_col)
                
            else:
                # Numerical target - use colormap
                scatter = plt.scatter(
                    pca_df['PC1'], pca_df['PC2'],
                    c=original_df[target_col], alpha=0.7, s=10,
                    cmap='viridis'
                )
                plt.colorbar(scatter, label=target_col)
            
            plt.title(f'PCA Colored by {target_col}')
            plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/pca_by_{target_col}.png", dpi=300)
            plt.close()
            
    # If period is available, create a period-colored plot
    if 'true_period' in original_df.columns:
        plt.figure(figsize=(12, 10))
        # Log scale for period is often better
        scatter = plt.scatter(
            pca_df['PC1'], pca_df['PC2'],
            c=np.log10(np.clip(original_df['true_period'].values, 0.01, None)), 
            alpha=0.7, s=10, cmap='plasma'
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label('log10(Period) [days]')
        plt.title('PCA Colored by Period')
        plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_by_period.png", dpi=300)
        plt.close()
        
    # If amplitude is available, create an amplitude-colored plot
    if 'true_amplitude' in original_df.columns:
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            pca_df['PC1'], pca_df['PC2'],
            c=np.log10(np.clip(original_df['true_amplitude'].values, 0.001, None)), 
            alpha=0.7, s=10, cmap='inferno'
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label('log10(Amplitude)')
        plt.title('PCA Colored by Amplitude')
        plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_by_amplitude.png", dpi=300)
        plt.close()

def save_pca_data(pca_df, original_df, pca_model, output_file='pca_results.csv'):
    """
    Save PCA results, including component values and loadings
    """
    # Combine PCA components with original data
    result_df = original_df.copy()
    for col in pca_df.columns:
        result_df[col] = pca_df[col].values
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Saved PCA results to {output_file}")
    
    # Save feature contributions/loadings
    loadings = pd.DataFrame(
        pca_model.components_.T,
        columns=[f'PC{i+1}' for i in range(pca_model.components_.shape[0])],
        index=original_df.columns
    )
    loadings.to_csv(output_file.replace('.csv', '_loadings.csv'))
    print(f"Saved feature loadings to {output_file.replace('.csv', '_loadings.csv')}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run PCA on PRIMVS data')
    parser.add_argument('input_file', help='Input FITS or CSV file')
    parser.add_argument('--output-dir', default='./figures', help='Output directory for plots')
    parser.add_argument('--output-file', default='pca_results.csv', help='Output file for PCA results')
    parser.add_argument('--n-components', type=int, default=10, help='Number of PCA components to calculate')
    parser.add_argument('--scale-method', choices=['robust', 'standard'], default='robust', 
                        help='Scaling method to use')
    parser.add_argument('--impute-method', choices=['knn', 'median'], default='knn',
                        help='Method to impute missing values')
    parser.add_argument('--sample-size', type=int, default=None, 
                        help='Randomly sample N records (useful for large datasets)')
    parser.add_argument('--feature-set', choices=['all', 'physical', 'embeddings'], default='all',
                        help='Which feature set to use for PCA')
    
    args = parser.parse_args()
    
    # Load data
    df = load_data(args.input_file)
    
    # Sample if requested (for large datasets)
    if args.sample_size and args.sample_size < len(df):
        df = df.sample(args.sample_size, random_state=42)
        print(f"Sampled {args.sample_size} records")
    
    # Identify features in the data
    feature_groups = identify_feature_columns(df)
    
    # Filter features based on user preference
    if args.feature_set == 'physical':
        feature_groups['embeddings'] = []
        feature_groups['additional'] = []
    elif args.feature_set == 'embeddings':
        feature_groups['physical'] = []
        feature_groups['additional'] = []
    
    # Prepare data
    df_scaled, df_processed = prepare_data(
        df, feature_groups, 
        scale_method=args.scale_method,
        impute_method=args.impute_method
    )
    
    # Apply PCA
    pca_df, pca_model = apply_pca(df_scaled, n_components=args.n_components)
    
    # Visualize results
    visualize_pca(pca_df, df, feature_groups, pca_model, output_dir=args.output_dir)
    
    # Save results
    save_pca_data(pca_df, df, pca_model, output_file=args.output_file)
    
    print("PCA analysis completed successfully!")

if __name__ == "__main__":
    main()
