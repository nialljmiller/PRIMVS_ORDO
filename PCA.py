#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
import seaborn as sns
from astropy.io import fits
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

def prepare_data(df, features, scale_method='robust'):
    """
    Prepare data for PCA:
    1. Filter to only the selected features
    2. Handle missing values
    3. Scale features
    4. Transform Period to log scale
    """
    # Filter to only requested features
    available_features = [f for f in features if f in df.columns]
    print(f"Using {len(available_features)} features: {available_features}")
    
    # Create a copy with only needed columns
    df_selected = df[available_features].copy()
    
    # Handle missing values and infinities
    df_selected.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Impute missing values with KNN
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_selected),
        columns=df_selected.columns
    )
    
    # Scale features
    if scale_method == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
    
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_imputed),
        columns=df_imputed.columns
    )
    
    # Transform period to log scale if present
    if 'true_period' in df_scaled.columns:
        # Add a small constant to avoid log(0)
        min_positive = df_imputed['true_period'][df_imputed['true_period'] > 0].min()
        period_data = df_imputed['true_period'].values
        period_data[period_data <= 0] = min_positive / 10
        
        # Apply log transform and scale
        period_col = df_scaled.columns.get_loc('true_period')
        df_scaled.iloc[:, period_col] = scaler.fit_transform(
            np.log10(period_data).reshape(-1, 1)
        )
    
    return df_scaled

def apply_pca(data, n_components=2):
    """
    Apply PCA to reduce dimensionality
    """
    # Initialize PCA
    pca = PCA(n_components=n_components, random_state=42)
    
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
    
    print(f"Explained variance: {explained_variance[:2]}")
    print(f"Cumulative variance: {cumulative_variance[:2]}")
    
    return pca_df, pca

def visualize_pca(pca_df, original_df, pca_model, output_dir='./figures'):
    """
    Create visualizations of the PCA results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Scatter plot of PC1 vs PC2
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
    
    # Plot 3: 2D histogram
    plt.figure(figsize=(10, 8))
    h = plt.hist2d(pca_df['PC1'], pca_df['PC2'], bins=100, cmap='viridis', norm=LogNorm())
    plt.colorbar(h[3], label='Count (log scale)')
    plt.title('2D Histogram of First Two Principal Components')
    plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_histogram.png", dpi=300)
    plt.close()
    
    # Plot 4: Scree plot (explained variance)
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
    
    # Plot 5: Feature loadings
    plt.figure(figsize=(12, 8))
    loadings = pd.DataFrame(
        pca_model.components_.T,
        columns=[f'PC{i+1}' for i in range(pca_model.components_.shape[0])],
        index=pca_model.feature_names_in_
    )
    
    # Only show top features
    top_features = pd.concat([
        loadings['PC1'].abs().sort_values(ascending=False),
        loadings['PC2'].abs().sort_values(ascending=False)
    ]).index.unique()[:15]
    
    sns.heatmap(loadings.loc[top_features, ['PC1', 'PC2']], 
                annot=True, cmap='coolwarm', fmt=".2f", center=0)
    plt.title('Feature Contributions to PC1 and PC2')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/pca_feature_contributions.png", dpi=300)
    plt.close()
    
    # If best_fap is available, create a FAP-colored plot
    if 'best_fap' in original_df.columns:
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(
            pca_df['PC1'], pca_df['PC2'],
            c=original_df['best_fap'], 
            alpha=0.7, s=10, cmap='viridis_r'  # Reversed so low FAP (better) is darker
        )
        cbar = plt.colorbar(scatter)
        cbar.set_label('False Alarm Probability (FAP)')
        plt.title('PCA Colored by False Alarm Probability')
        plt.xlabel(f'PC1 ({pca_model.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca_model.explained_variance_ratio_[1]:.2%} variance)')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/pca_by_fap.png", dpi=300)
        plt.close()
        
    # If period is available, create a period-colored plot
    if 'true_period' in original_df.columns:
        plt.figure(figsize=(12, 10))
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
    Save PCA results to file
    """
    # Add PCA components to original data for saving
    result_df = original_df.copy()
    for col in pca_df.columns:
        result_df[col] = pca_df[col].values
    
    # Save to CSV
    result_df.to_csv(output_file, index=False)
    print(f"Saved PCA results to {output_file}")
    
    # Save feature loadings
    loadings = pd.DataFrame(
        pca_model.components_.T,
        columns=[f'PC{i+1}' for i in range(pca_model.components_.shape[0])],
        index=pca_model.feature_names_in_
    )
    loadings.to_csv(output_file.replace('.csv', '_loadings.csv'))

def main():
    # Hard-coded parameters
    input_file = '../PRIMVS_CC.fits'  # Change this to your input file path
    output_dir = './figures'
    output_file = 'pca_results.csv'
    n_components = 10
    sample_size = 100000  # Set to None to use all data
    
    # List of features to use (from Table 4 in PRIMVS paper + the key metrics you wanted)
    features = [
        # Key metrics specifically requested
        "best_fap", "true_period", "true_amplitude",
        
        # Color features
        "z_med_mag-ks_med_mag", "y_med_mag-ks_med_mag", 
        "j_med_mag-ks_med_mag", "h_med_mag-ks_med_mag",
        
        # Positional features
        "l", "b", 
        
        # Statistical features
        "Cody_M", "stet_k", "eta_e", "med_BRP", "range_cum_sum",
        "max_slope", "MAD", "mean_var", "percent_amp",
        "roms", "p_to_p_var", "lag_auto", "AD", "std_nxs",
        
        # Basic statistics
        "skew", "kurt",
        
        # Embedding features (contrastive learning dimensions)
        # Adding the first 30 embedding dimensions if available
        #*[str(i) for i in range(30)]
    ]
    
    # Load data
    df = load_data(input_file)
    
    # Sample if needed (for large datasets)
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)
        print(f"Sampled {sample_size} records")
    
    # Prepare data (filter, impute, scale)
    df_scaled = prepare_data(df, features)
    
    # Apply PCA
    pca_df, pca_model = apply_pca(df_scaled, n_components=n_components)
    
    # Visualize results
    visualize_pca(pca_df, df, pca_model, output_dir=output_dir)
    
    # Save results
    save_pca_data(pca_df, df, pca_model, output_file=output_file)
    
    print("PCA analysis completed successfully!")

if __name__ == "__main__":
    main()
