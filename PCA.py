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
    Prepare data for PCA with advanced feature-specific preprocessing,
    automatic percentile capping, and correlation handling
    """
    # Filter to only requested features
    available_features = [f for f in features if f in df.columns]
    print(f"Using {len(available_features)} features initially: {available_features}")
    
    # Create a copy with only needed columns
    df_selected = df[available_features].copy()
    
    # Handle missing values and infinities
    df_selected.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # First, impute missing values with KNN
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df_selected),
        columns=df_selected.columns
    )
    
    # ===== 1. PERCENTILE CAPPING =====
    # Cap values at 1st and 99th percentiles to handle extreme outliers
    print("\nPerforming percentile capping (1-99):")
    df_capped = df_imputed.copy()
    
    for feature in df_capped.columns:
        # Skip embedding features (dimensions from contrastive learning)
        if feature.isdigit():
            continue
            
        # Calculate 1st and 99th percentiles
        p01 = np.nanpercentile(df_capped[feature].values, 1)
        p99 = np.nanpercentile(df_capped[feature].values, 99)
        
        # Count values outside this range
        n_outliers = ((df_capped[feature] < p01) | (df_capped[feature] > p99)).sum()
        if n_outliers > 0:
            # Cap values
            df_capped[feature] = df_capped[feature].clip(p01, p99)
            pct_outliers = 100 * n_outliers / len(df_capped)
            print(f"  Capped {feature}: {n_outliers} outliers ({pct_outliers:.2f}%)")
    
    # ===== 2. FEATURE-SPECIFIC TRANSFORMATIONS =====
    print("\nApplying feature-specific transformations:")
    df_transformed = df_capped.copy()
    
    for feature in df_transformed.columns:
        data = df_transformed[feature].values
        
        # Skip embedding features (dimensions from contrastive learning)
        if feature.isdigit():
            continue
            
        # Calculate skewness
        skewness = pd.Series(data).skew()
        
        # Check for large range (max/min ratio)
        feature_range = np.nanmax(data) / (np.nanmin(data) + 1e-10)  # Avoid division by zero
        
        # Identify transformation method
        if feature == 'true_period':
            # Log transform for period (astronomical convention)
            min_positive = max(np.nanmin(data[data > 0]), 1e-10)
            data[data <= 0] = min_positive / 10
            data = np.log10(data)
            print(f"  Applied log10 transform to {feature} (astronomical convention)")
            
        elif feature == 'true_amplitude':
            # Log transform for amplitude (astronomical convention)
            min_positive = max(np.nanmin(data[data > 0]), 1e-10)
            data[data <= 0] = min_positive / 10
            data = np.log10(data)
            print(f"  Applied log10 transform to {feature} (astronomical convention)")
            
        elif feature == 'best_fap':
            # FAP is already between 0-1, but log transform can help highlight differences
            # Add small constant to avoid log(0)
            min_positive = max(np.nanmin(data[data > 0]), 1e-10)
            data[data <= 0] = min_positive / 10
            data = -np.log10(data)  # Negative log transform makes smaller FAP values larger (better)
            print(f"  Applied -log10 transform to {feature} (higher values = better significance)")
            
        elif abs(skewness) > 2:
            # Highly skewed - use log transform
            # Check if all values are positive
            if np.nanmin(data) <= 0:
                # Add offset to make all values positive
                offset = abs(np.nanmin(data)) + 1e-10
                data = np.log1p(data + offset)
                print(f"  Applied log1p transform with offset to {feature} (skewness: {skewness:.2f})")
            else:
                data = np.log1p(data)
                print(f"  Applied log1p transform to {feature} (skewness: {skewness:.2f})")
                
        elif abs(skewness) > 1:
            # Moderately skewed - use square root or cube root
            if np.nanmin(data) < 0:
                # Use cube root for negative values
                data = np.cbrt(data)
                print(f"  Applied cube root transform to {feature} (skewness: {skewness:.2f})")
            else:
                # Use square root for positive values
                data = np.sqrt(np.abs(data)) * np.sign(data)
                print(f"  Applied square root transform to {feature} (skewness: {skewness:.2f})")
                
        elif feature_range > 1000:
            # Very large range - use log transform
            if np.nanmin(data) <= 0:
                offset = abs(np.nanmin(data)) + 1e-10
                data = np.log1p(data + offset)
                print(f"  Applied log1p transform to {feature} (large range: {feature_range:.1e})")
            else:
                data = np.log1p(data)
                print(f"  Applied log1p transform to {feature} (large range: {feature_range:.1e})")
        
        # Update the transformed DataFrame
        df_transformed[feature] = data
    
    # ===== 3. CORRELATION HANDLING =====
    # Automatically detect and remove highly correlated features
    corr_threshold = 0.85  # Correlation threshold for considering features as redundant
    
    # Calculate correlation matrix (excluding embedding features)
    non_embedding_cols = [col for col in df_transformed.columns if not col.isdigit()]
    corr_matrix = df_transformed[non_embedding_cols].corr().abs()
    
    # Create a mask for the upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = []
    
    print("\nChecking for correlated features (threshold: 0.85):")
    for col in upper.columns:
        # Find correlated features for this column
        correlated = upper[col][upper[col] > corr_threshold].index.tolist()
        if correlated:
            # Calculate variance for each feature to decide which to keep
            variances = {feature: df_transformed[feature].var() for feature in [col] + correlated}
            variances_sorted = sorted(variances.items(), key=lambda x: x[1], reverse=True)
            
            # Keep the feature with highest variance, add others to drop list
            keep_feature = variances_sorted[0][0]
            drop_features = [f for f in [col] + correlated if f != keep_feature and f not in to_drop]
            
            if drop_features:
                for f in drop_features:
                    corr_val = upper[col][f] if f in upper[col] else upper[f][col]
                    print(f"  {f} and {keep_feature} are correlated (r={corr_val:.3f}). Keeping {keep_feature} (higher variance)")
                to_drop.extend(drop_features)
    
    # Remove correlated features
    if to_drop:
        print(f"  Removing {len(to_drop)} correlated features: {to_drop}")
        df_transformed = df_transformed.drop(columns=to_drop)
    else:
        print("  No highly correlated features found")
    
    # ===== 4. FINAL SCALING =====
    # Apply final scaling
    if scale_method == 'robust':
        scaler = RobustScaler(quantile_range=(5, 95))  # Use 5-95 percentile range to be more robust
        print("\nUsing robust scaling with 5-95 percentile range")
    else:
        scaler = StandardScaler()
        print("\nUsing standard scaling (z-score normalization)")
    
    # Apply scaling to all features
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_transformed),
        columns=df_transformed.columns
    )
    
    # Remove any remaining infinities or NaNs
    df_scaled.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # If any NaNs remain, fill with 0 (mean of scaled data)
    df_scaled.fillna(0, inplace=True)
    
    # Print final feature count
    print(f"\nFinal feature set: {len(df_scaled.columns)} features")
    
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
    input_file = '../PRIMVS/PRIMVS_P.fits'  # Change this to your input file path
    output_dir = './pca_figures'
    output_file = 'pca_results.csv'
    n_components = 10
    sample_size = None#100000  # Set to None to use all data
    
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
