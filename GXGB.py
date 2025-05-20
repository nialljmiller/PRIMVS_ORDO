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
    period = ["true_period", "ls_fap", "pdm_fap", "ce_fap", "gp_fap"]
    periodograms = ["ls_y_y_0", "ls_peak_width_0", "pdm_y_y_0", "pdm_peak_width_0", "ce_y_y_0", "ce_peak_width_0"]
    quality = ["chisq", "uwe"]
    coords = ["l", "b", "parallax", "pmra", "pmdec"]
    embeddings = [str(i) for i in range(128)]

    full_set = variability + colour + mags + period + periodograms + quality + coords + embeddings
    common = set(train.columns).intersection(test.columns)
    usable = [f for f in full_set if f in common]
    print(f"Using {len(usable)} of {len(full_set)} total features")
    return usable


# Add this at the top of your file with other imports
def get_gpu_count():
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '-L'], stdout=subprocess.PIPE)
        return len(result.stdout.decode('utf-8').strip().split('\n'))
    except:
        try:
            # Alternative method using pycuda
            import pycuda.driver as cuda
            cuda.init()
            return cuda.Device.count()
        except:
            return 1  # Default to 1 if detection fails


def train_xgb(train_df, test_df, features, label_col, out_file):
    # Preprocess
    X_train = train_df[features].replace([np.inf, -np.inf], np.nan)
    y = LabelEncoder().fit_transform(train_df[label_col])
    X_test = test_df[features].replace([np.inf, -np.inf], np.nan)

    dtrain = xgb.DMatrix(X_train, label=y)
    dtest = xgb.DMatrix(X_test)

    # Handle class imbalance
    weights = len(y) / (np.bincount(y) * len(np.bincount(y)))



    # In your train_xgb function, replace the params block with this:
    num_gpus = get_gpu_count()
    print(f"Auto-detected {num_gpus} GPUs")

    params = {
        'objective': 'multi:softprob',
        'num_class': len(np.unique(y)),
        'eval_metric': 'mlogloss',
        'learning_rate': 0.01,  # Lower learning rate for higher rounds
        'max_depth': 15,        # Increased from 12
        'min_child_weight': 30,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'max_bin': 1024         # Higher precision binning
    }
    
    # Auto-configure GPU usage
    if num_gpus > 1:
        params['n_gpus'] = num_gpus
    else:
        params['gpu_id'] = 0    # Use gpu_id only when single GPU



    print("Training...")
    model = xgb.train(params, dtrain, num_boost_round=2000, evals=[(dtrain, 'train')], verbose_eval=50)

    # Predict
    probs = model.predict(dtest)
    preds = np.argmax(probs, axis=1)
    confs = np.max(probs, axis=1)

    # Decode labels
    label_encoder = LabelEncoder().fit(train_df[label_col])
    pred_labels = label_encoder.inverse_transform(preds)

    # Save output
    test_df = test_df.copy()
    test_df['xgb_predicted_class_id'] = preds
    test_df['xgb_predicted_class'] = pred_labels
    test_df['xgb_confidence'] = confs
    test_df.to_csv(out_file, index=False)
    print(f"Saved predictions to {out_file}")

    # Importance
    imp = model.get_score(importance_type='gain')
    top_feats = sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20]
    print("\nTop features:")
    for f, score in top_feats:
        print(f"  {f}: {score:.3f}")

    # Visuals
    plot_period_amplitude(test_df, "xgb_predicted_class")
    plot_galactic_distribution(test_df, "xgb_predicted_class")
    plot_color_color(test_df, "xgb_predicted_class")
    plot_astronomical_map(test_df, "xgb_predicted_class")
    plot_hr_diagram(test_df, "xgb_predicted_class")

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
