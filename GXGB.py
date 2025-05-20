import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from astropy.io import fits
from vis import *

def load_fits_to_df(path):
    # Read a FITS file into a DataFrame
    print(f"Loading {path}...")
    with fits.open(path) as hdul:
        return pd.DataFrame(hdul[1].data)

def get_feature_list(train, test):
    # Basic feature sets we’ve previously used
    variability = ["MAD", "eta", "eta_e", "true_amplitude", "mean_var", "std_nxs", "range_cum_sum", "max_slope", "percent_amp", "stet_k", "roms", "lag_auto", "Cody_M", "AD", "med_BRP", "p_to_p_var"]
    colour = ["z_med_mag-ks_med_mag", "y_med_mag-ks_med_mag", "j_med_mag-ks_med_mag", "h_med_mag-ks_med_mag"]
    mags = ["ks_med_mag", "ks_mean_mag", "ks_std_mag", "ks_mad_mag", "j_med_mag", "h_med_mag"]
    period = ["true_period"]#, "ls_fap", "pdm_fap", "ce_fap", "gp_fap"]
    periodograms = []#"ls_y_y_0", "ls_peak_width_0", "pdm_y_y_0", "pdm_peak_width_0", "ce_y_y_0", "ce_peak_width_0"]
    quality = ["chisq", "uwe"]
    coords = ["l", "b", "parallax", "pmra", "pmdec"]
    embeddings = [str(i) for i in range(128)]

    full_set = variability + colour + mags + period + periodograms + quality + coords + embeddings
    common = set(train.columns).intersection(test.columns)
    usable = [f for f in full_set if f in common]
    print(f"Using {len(usable)} of {len(full_set)} total features")
    return usable



# Add at top with imports
import subprocess
from sklearn.model_selection import train_test_split

# Auto-detect GPU count
def get_gpu_count():
    try:
        result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE)
        return len(result.stdout.decode('utf-8').strip().split('\n'))
    except:
        return 1







def train_xgb(train_df, test_df, features, label_col, out_file):
    """
    Enhanced XGBoost training with improved preprocessing, class balancing, and overfitting control.
    """
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.impute import SimpleImputer
    import numpy as np
    
    # === IMPROVED PREPROCESSING ===
    print("Applying enhanced preprocessing...")
    
    # Step 1: Replace infinities and prepare data
    X_train = train_df[features].replace([np.inf, -np.inf], np.nan)
    X_test = test_df[features].replace([np.inf, -np.inf], np.nan)
    
    # Step 2: Get statistics from training data only
    train_quantiles = X_train.quantile([0.001, 0.999])
    train_min = train_quantiles.iloc[0]
    train_max = train_quantiles.iloc[1]
    
    # Step 3: Apply clipping to both datasets to handle outliers
    X_train = X_train.clip(lower=train_min, upper=train_max, axis=1)
    X_test = X_test.clip(lower=train_min, upper=train_max, axis=1)
    
    # Step 4: Intelligent imputation (median for each feature)
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    
    # Step 5: Standardize features (RobustScaler handles outliers better)
    scaler = RobustScaler()  # More robust to outliers than StandardScaler
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Convert back to DataFrames for easier handling
    X_train = pd.DataFrame(X_train_scaled, columns=features, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=features, index=X_test.index)
    
    # === IMPROVED CLASS HANDLING ===
    # Process labels
    label_encoder = LabelEncoder().fit(train_df[label_col])
    y = label_encoder.transform(train_df[label_col])
    
    # Create validation split (handle rare classes)
    class_counts = np.bincount(y)
    if np.min(class_counts) < 2:
        print("Found rare classes - using non-stratified validation split")
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y, test_size=0.2, random_state=42)
    else:
        X_train_main, X_val, y_train_main, y_val = train_test_split(
            X_train, y, test_size=0.2, random_state=42, stratify=y)
    
    # Better class weighting - smoother weight scale
    class_weights = {}
    for i, count in enumerate(class_counts):
        # Square root scaling for softer penalties on rare classes
        class_weights[i] = np.sqrt(np.sum(class_counts) / (len(class_counts) * count))
    
    print("Class weights:", {label_encoder.inverse_transform([i])[0]: f"{w:.2f}" 
                            for i, w in class_weights.items()})
    
    # Create sample weights for training data
    sample_weights = np.array([class_weights[y_i] for y_i in y_train_main])
    
    # === IMPROVED MODEL TRAINING ===
    # Convert to XGBoost format with sample weights
    dtrain = xgb.DMatrix(X_train_main, label=y_train_main, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    dtest = xgb.DMatrix(X_test)
    
    # Auto-detect GPUs
    num_gpus = get_gpu_count()
    print(f"Auto-detected {num_gpus} GPUs for training")
    
    # Optimized parameters with overfitting controls
    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'eval_metric': 'mlogloss',
        'learning_rate': 0.01,
        'max_depth': 10,              # Reduced to help with overfitting
        'min_child_weight': 5,        # Reduced for better balance
        'subsample': 0.7,             # Reduced to prevent overfitting
        'colsample_bytree': 0.7,      # Column subsampling reduces overfitting
        'colsample_bylevel': 0.7,     # Helps with high-dimensional data
        'reg_alpha': 0.1,             # L1 regularization 
        'reg_lambda': 1.0,            # L2 regularization
        'tree_method': 'hist',        # Works on both CPU and GPU
        'max_bin': 256,               # Reduced for faster training
        'grow_policy': 'lossguide'
    }
    
    # Configure GPU efficiently
    if num_gpus > 0:
        params['device'] = 'cuda' if num_gpus > 1 else 'cuda:0'
    
    # Store evaluation results
    evals_result = {}
    
    # Train with early stopping
    print("Training model with anti-overfitting parameters...")
    model = xgb.train(
        params, 
        dtrain,
        num_boost_round=30000,
        early_stopping_rounds=200,    # More patience
        evals=[(dtrain, 'train'), (dval, 'validation')],
        evals_result=evals_result,
        verbose_eval=50
    )
    
    # === PREDICTION AND OUTPUTS ===
    best_iter = model.best_iteration
    print(f"Best iteration: {best_iter}")
    
    # Get predictions using best iteration
    probs = model.predict(dtest, iteration_range=(0, best_iter+1))
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    
    # Decode predictions
    pred_labels = label_encoder.inverse_transform(preds)
    
    # Save results
    test_df_result = test_df.copy()
    test_df_result['xgb_predicted_class_id'] = preds
    test_df_result['xgb_predicted_class'] = pred_labels
    test_df_result['xgb_confidence'] = confs
    test_df_result.to_csv(out_file, index=False)
    print(f"Saved predictions to {out_file}")
    test_df = test_df_result.copy()

    # Feature importance analysis
    importance = model.get_score(importance_type='gain')
    top_feats = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\nTop features:")
    for f, score in top_feats:
        print(f"  {f}: {score:.3f}")
    

    plot_period_amplitude(test_df, "xgb_predicted_class")
    plot_galactic_distribution(test_df, "xgb_predicted_class")
    plot_color_color(test_df, "xgb_predicted_class")
    plot_astronomical_map(test_df, "xgb_predicted_class")
    plot_hr_diagram(test_df, "xgb_predicted_class")

    plot_xgb_training_loss(evals_result)

    plot_bailey_diagram(test_df, "xgb_predicted_class")
    plot_galactic_coords(test_df, "xgb_predicted_class")
    plot_confidence_entropy(test_df, "xgb_predicted_class")
    plot_xgb_class_probability_heatmap(probs, label_encoder.classes_)
    plot_xgb_top2_confidence_scatter(probs, preds, label_encoder.classes_)

    #things that require extra shit
    feat_names = [x[0] for x in top_feats]
    scores = [x[1] for x in top_feats]
    plot_xgb_feature_importance(feat_names, scores)
    plot_confidence_distribution(confs, preds, label_encoder.classes_)
    top_feats = features[:12] if len(features) > 12 else features
    plot_feature_class_correlations(test_df, top_feats, "xgb_predicted_class")

    # If ground truth exists in test
    if label_col in test_df:
        y_true = test_df[label_col]
        y_pred = test_df['xgb_predicted_class']
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred))
        plot_classification_performance(y_true, y_pred, label_encoder.classes_)
        plot_misclassification_analysis(y_true, y_pred, probs, label_encoder.classes_)

    return model

def main():
    # Inputs from CLI or defaults
    if len(sys.argv) < 3:
        train_path = "../PRIMVS/PRIMVS_P_GAIA.fits"
        test_path = "../PRIMVS/PRIMVS_P.fits"
        out_file = "xgb_predictions.csv"
        print("Using default file paths")
    else:
        train_path, test_path = sys.argv[1:3]
        out_file = sys.argv[3] if len(sys.argv) > 3 else "xgb_predictions.csv"

    train_df = load_fits_to_df(train_path)
    test_df = load_fits_to_df(test_path)

    label = "best_class_name"
    print(train_df[label].value_counts())

    feats = get_feature_list(train_df, test_df)
    train_xgb(train_df, test_df, feats, label, out_file)

if __name__ == "__main__":
    main()
