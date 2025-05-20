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
from vis import visualize_pca

def load_data(filepath):
    # Load a FITS table or CSV into a DataFrame
    print(f"Loading data from {filepath}...")
    if filepath.endswith('.fits'):
        with fits.open(filepath) as hdul:
            df = pd.DataFrame(hdul[1].data)
    else:
        df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} records with {len(df.columns)} features")
    return df

def prepare_data(df, features, scale_method='robust'):
    # Select only desired columns
    features = [f for f in features if f in df.columns]
    print(f"Using {len(features)} features: {features}")
    df = df[features].copy()

    # Replace infs with NaNs then impute using KNN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = pd.DataFrame(KNNImputer(n_neighbors=5).fit_transform(df), columns=df.columns)

    # Clip extreme values (1st to 99th percentile)
    print("\nApplying percentile clipping...")
    for f in df.columns:
        if f.isdigit(): continue
        p01, p99 = np.nanpercentile(df[f], 1), np.nanpercentile(df[f], 99)
        df[f] = df[f].clip(p01, p99)

    # Apply various transformations depending on skew/range
    print("\nTransforming features...")
    for f in df.columns:
        x = df[f].values
        if f.isdigit(): continue
        sk = pd.Series(x).skew()
        rng = np.nanmax(x) / (np.nanmin(x) + 1e-10)

        if f == 'true_period' or f == 'true_amplitude':
            x[x <= 0] = max(np.nanmin(x[x > 0]), 1e-10) / 10
            x = np.log10(x)
            print(f"  log10({f})")
        elif f == 'best_fap':
            x[x <= 0] = max(np.nanmin(x[x > 0]), 1e-10) / 10
            x = -np.log10(x)
            print(f"  -log10({f})")
        elif abs(sk) > 2:
            offset = abs(np.nanmin(x)) + 1e-10 if np.nanmin(x) <= 0 else 0
            x = np.log1p(x + offset)
            print(f"  log1p({f})")
        elif abs(sk) > 1:
            x = np.cbrt(x) if np.nanmin(x) < 0 else np.sqrt(np.abs(x)) * np.sign(x)
            print(f"  root transform ({f})")
        elif rng > 1000:
            offset = abs(np.nanmin(x)) + 1e-10 if np.nanmin(x) <= 0 else 0
            x = np.log1p(x + offset)
            print(f"  log1p(range fix: {f})")
        df[f] = x

    # Drop redundant features (high correlation)
    print("\nChecking correlations...")
    keep = df.columns[~df.columns.str.isdigit()]
    corr = df[keep].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = []
    for col in upper.columns:
        hits = upper[col][upper[col] > 0.85].index.tolist()
        if hits:
            all_feats = [col] + hits
            var = {k: df[k].var() for k in all_feats}
            top = max(var, key=var.get)
            to_drop += [k for k in all_feats if k != top and k not in to_drop]
    df.drop(columns=to_drop, inplace=True)
    if to_drop:
        print(f"  Dropped: {to_drop}")

    # Final scaling
    scaler = RobustScaler(quantile_range=(5, 95)) if scale_method == 'robust' else StandardScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    df.fillna(0, inplace=True)
    print(f"\nFinal feature count: {len(df.columns)}")
    return df

def apply_pca(data, n_components=2):
    # Simple PCA wrapper
    pca = PCA(n_components=n_components, random_state=42)
    comps = pca.fit_transform(data)
    df = pd.DataFrame(comps, columns=[f'PC{i+1}' for i in range(n_components)])
    print(f"Explained variance: {pca.explained_variance_ratio_[:2]}")
    return df, pca




def save_pca_data(pca_df, original_df, pca_model, output_file='pca_results.csv'):
    # Merge and save PCA + loadings
    df_out = original_df.copy()
    for col in pca_df.columns:
        df_out[col] = pca_df[col]
    df_out.to_csv(output_file, index=False)

    load = pd.DataFrame(pca_model.components_.T, columns=[f'PC{i+1}' for i in range(pca_model.n_components_)], index=pca_model.feature_names_in_)
    load.to_csv(output_file.replace('.csv', '_loadings.csv'))




def main():
    input_file = '../PRIMVS/PRIMVS_P.fits'
    output_dir = './pca_figures'
    output_file = 'pca_results.csv'
    n_components = 10
    sample_size = None

    features = [
        "best_fap", "true_period", "true_amplitude",
        "z_med_mag-ks_med_mag", "y_med_mag-ks_med_mag", 
        "j_med_mag-ks_med_mag", "h_med_mag-ks_med_mag",
        "l", "b",
        "Cody_M", "stet_k", "eta_e", "med_BRP", "range_cum_sum",
        "max_slope", "MAD", "mean_var", "percent_amp",
        "roms", "p_to_p_var", "lag_auto", "AD", "std_nxs",
        "skew", "kurt"
    ]

    df = load_data(input_file)
    if sample_size and sample_size < len(df):
        df = df.sample(sample_size, random_state=42)

    df_scaled = prepare_data(df, features)
    pca_df, pca_model = apply_pca(df_scaled, n_components=n_components)
    visualize_pca(pca_df, df, pca_model, output_dir)
    save_pca_data(pca_df, df, pca_model, output_file)
    print("Done.")

if __name__ == "__main__":
    main()
