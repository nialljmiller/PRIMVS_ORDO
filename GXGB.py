import sys
import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from astropy.io import fits
import matplotlib.pyplot as plt
import seaborn as sns

# === Load FITS Files ===
def load_fits_to_dataframe(fits_path):
    print(f"Loading data from {fits_path}...")
    with fits.open(fits_path) as hdul:
        data = hdul[1].data
        try:
            return pd.DataFrame(data)
        except Exception as e:
            print(f"Direct conversion failed: {e}")
            try:
                return pd.DataFrame.from_records(data)
            except Exception as e:
                print(f"Structured array approach failed: {e}")
                columns = data.names
                df = pd.DataFrame()
                for col in columns:
                    df[col] = data[col]
                return df

# === Feature Selection ===
def select_features(train_df, test_df):
    # Define feature categories
    VARIABILITY_FEATURES = [
        "MAD", "eta", "eta_e", "true_amplitude", "mean_var", "std_nxs", 
        "range_cum_sum", "max_slope", "percent_amp", "stet_k", "roms", 
        "lag_auto", "Cody_M", "AD", "med_BRP", "p_to_p_var"
    ]
    
    COLOR_FEATURES = [
        "z_med_mag-ks_med_mag", "y_med_mag-ks_med_mag", 
        "j_med_mag-ks_med_mag", "h_med_mag-ks_med_mag"
    ]
    
    MAGNITUDE_FEATURES = [
        "ks_med_mag", "ks_mean_mag", "ks_std_mag", "ks_mad_mag",
        "j_med_mag", "h_med_mag"
    ]
    
    PERIOD_FEATURES = [
        "true_period", "ls_fap", "pdm_fap", "ce_fap", "gp_fap"
    ]
    
    PERIODOGRAM_FEATURES = [
        "ls_y_y_0", "ls_peak_width_0", "pdm_y_y_0", "pdm_peak_width_0", 
        "ce_y_y_0", "ce_peak_width_0"
    ]
    
    QUALITY_FEATURES = [
        "chisq", "uwe"
    ]
    
    POSITION_FEATURES = [
        "l", "b", "parallax", "pmra", "pmdec"
    ]
    
    # Combine all feature groups
    ALL_FEATURES = (VARIABILITY_FEATURES + COLOR_FEATURES + MAGNITUDE_FEATURES + 
                   PERIOD_FEATURES + PERIODOGRAM_FEATURES + QUALITY_FEATURES + 
                   POSITION_FEATURES)
    
    # Filter for features present in both datasets
    train_features = set(train_df.columns)
    test_features = set(test_df.columns)
    common_features = list(train_features.intersection(test_features))
    available_features = [f for f in ALL_FEATURES if f in common_features]
    
    print(f"Using {len(available_features)} features from {len(ALL_FEATURES)} candidates")
    print(f"Features: {available_features}")
    
    return available_features

# === Model Training and Prediction ===
def train_xgboost_gpu(train_df, test_df, features, label_col, output_file):
    # Prepare data
    X_train = train_df[features].replace([np.inf, -np.inf], np.nan)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(train_df[label_col])
    
    X_test = test_df[features].replace([np.inf, -np.inf], np.nan)
    
    # Create DMatrix - optimized data structure for XGBoost
    print("Converting data to DMatrix format...")
    dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
    dtest = xgb.DMatrix(X_test)
    
    # Calculate class weights for imbalanced data
    class_counts = np.bincount(y_train_encoded)
    total_samples = len(y_train_encoded)
    class_weights = total_samples / (len(class_counts) * class_counts)
    
    # Parameters optimized for astronomical classification with GPU
    params = {
        'objective': 'multi:softprob',
        'num_class': len(label_encoder.classes_),
        'eval_metric': 'mlogloss',
        'learning_rate': 0.05,
        'max_depth': 12,
        'min_child_weight': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'gpu_hist',       # Use GPU for training
        'predictor': 'gpu_predictor',    # Use GPU for prediction
        'gpu_id': 0,                     # Specify which GPU to use
        'max_bin': 256                   # Optimization for GPU
    }
    
    # Print class distribution and weights
    print("\nClass distribution:")
    for i, cls in enumerate(label_encoder.classes_):
        count = class_counts[i]
        weight = class_weights[i]
        print(f"  {cls}: {count} samples, weight: {weight:.3f}")
    
    # Train the model
    print(f"\nTraining XGBoost model on GPU with {len(features)} features...")
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,      # Increase for better performance if time permits
        evals=[(dtrain, 'train')],
        verbose_eval=50,          # Print evaluation every 50 rounds
    )
    
    # Run inference on GPU
    print(f"Running inference on {len(X_test)} samples...")
    probs = model.predict(dtest)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    
    # Convert predictions back to original labels
    pred_labels = label_encoder.inverse_transform(preds)
    
    # Add predictions to test DataFrame
    test_df_result = test_df.copy()
    test_df_result["xgb_predicted_class_id"] = preds
    test_df_result["xgb_predicted_class"] = pred_labels
    test_df_result["xgb_confidence"] = confs
    
    # Save results
    test_df_result.to_csv(output_file, index=False)
    print(f"\nSaved predictions to {output_file}")
    
    # Feature importance analysis
    importance = model.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 20 features by importance:")
    for feature, score in sorted_importance[:20]:
        print(f"  {feature}: {score:.4f}")
    
    # Create feature importance plot
    plt.figure(figsize=(12, 8))
    features = [x[0] for x in sorted_importance[:20]]
    scores = [x[1] for x in sorted_importance[:20]]
    sns.barplot(x=scores, y=features)
    plt.title('Top 20 Feature Importance (gain)')
    plt.tight_layout()
    plt.savefig('xgb_feature_importance.png', dpi=300)
    
    # Evaluate if ground truth is available
    if label_col in test_df_result.columns:
        print("\nEvaluation on test set:")
        y_true = test_df_result[label_col]
        y_pred = test_df_result["xgb_predicted_class"]
        print(classification_report(y_true, y_pred))
    
    return model, label_encoder

# === Main Function ===
def main():
    if len(sys.argv) < 3:
        print("Usage: python xgb_classify.py <training_fits_file> <testing_fits_file> [output_file]")
        print("Example: python xgb_classify.py primvs.fits primvs_gaia.fits predictions.csv")
        sys.exit(1)
    
    training_file = sys.argv[1]
    testing_file = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else "xgb_predictions.csv"
    
    # Load data
    train_df = load_fits_to_dataframe(training_file)
    print(f"Loaded training data: {len(train_df)} rows")
    
    test_df = load_fits_to_dataframe(testing_file)
    print(f"Loaded testing data: {len(test_df)} rows")
    
    # Analyze classes
    label_col = "best_class_name"  # Change if your label column is different
    print("\nExamining class distribution in training data:")
    train_classes = train_df[label_col].value_counts()
    print(f"Found {len(train_classes)} unique classes")
    print(train_classes.head(10))
    
    # Select features
    features = select_features(train_df, test_df)
    
    # Train and predict
    model, label_encoder = train_xgboost_gpu(train_df, test_df, features, label_col, output_file)
    
    print("\nProcessing complete!")

if __name__ == "__main__":
    main()