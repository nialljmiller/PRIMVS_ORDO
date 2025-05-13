import lightgbm as lgb
from sklearn.metrics import classification_report
import numpy as np

# Feature selection - expand to include more of your features
FEATURES = [
    # Color features
    "z_med_mag-ks_med_mag", "y_med_mag-ks_med_mag", "j_med_mag-ks_med_mag", "h_med_mag-ks_med_mag",
    # Positional features
    "l", "b", "parallax", "pmra", "pmdec",
    # Variability features
    "MAD", "eta", "eta_e", "true_amplitude", "mean_var", "std_nxs", "range_cum_sum", 
    "max_slope", "percent_amp", "stet_k", "roms", "lag_auto", "Cody_M", "AD",
    # Period features
    "true_period",
    # Magnitude features
    "ks_med_mag", "ks_std_mag", "ks_mad_mag",
    # Quality metrics
    "chisq", "uwe",
    # Periodogram statistics
    "ls_fap", "pdm_fap", "ce_fap", "gp_fap"
]

def train_lightgbm(train_df, test_df, features, label_col, output_file):
    # Prepare data
    X_train = train_df[features].replace([np.inf, -np.inf], np.nan)
    y_train = train_df[label_col]
    
    X_test = test_df[features].replace([np.inf, -np.inf], np.nan)
    
    # Calculate class weights to handle imbalance
    class_counts = y_train.value_counts()
    total_samples = len(y_train)
    class_weights = {i: total_samples / (len(class_counts) * count) 
                    for i, count in enumerate(class_counts)}
    
    # Create LightGBM dataset
    train_data = lgb.Dataset(
        X_train, 
        label=y_train,
        weight=y_train.map(lambda x: class_weights.get(x, 1.0))
    )
    
    # Parameters - tuned for astronomical classification
    params = {
        'objective': 'multiclass',
        'num_class': len(class_counts),
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 128,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'max_depth': 15,
        'min_data_in_leaf': 50,
        'num_iterations': 300,
        'verbose': 1,
        'force_col_wise': True  # More memory efficient
    }
    
    print(f"Training LightGBM model with {len(features)} features...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(50)]
    )
    
    # Inference
    print(f"Running inference on {len(X_test)} samples...")
    probs = model.predict(X_test)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)
    
    # Add predictions to output
    test_df_out = test_df.copy()
    test_df_out["lgbm_predicted_class_id"] = preds
    test_df_out["lgbm_confidence"] = confs
    
    # Decode class IDs back to names
    class_mapping = {i: label for i, label in enumerate(class_counts.index)}
    test_df_out["lgbm_predicted_class"] = [class_mapping[p] for p in preds]
    
    # Save results
    test_df_out.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")
    
    # Evaluate if ground truth is available
    if label_col in test_df_out.columns:
        y_true = test_df_out[label_col]
        y_pred = test_df_out["lgbm_predicted_class"]
        print(classification_report(y_true, y_pred))
    
    # Print feature importance
    importance = model.feature_importance(importance_type='gain')
    feature_importance = sorted(zip(features, importance), key=lambda x: x[1], reverse=True)
    print("\nTop 20 features by importance:")
    for feature, importance in feature_importance[:20]:
        print(f"{feature}: {importance}")
    
    return model, test_df_out