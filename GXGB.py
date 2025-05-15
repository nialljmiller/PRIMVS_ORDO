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
# Import visualization functions
from vis import (
    plot_classification_performance,
    plot_period_amplitude,
    plot_feature_class_correlations,
    plot_confidence_distribution,
    plot_xgb_feature_importance,
    plot_xgb_class_probability_heatmap,
    plot_xgb_top2_confidence_scatter,
    plot_astronomical_map,
    plot_misclassification_analysis
)

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

    # Include contrastive learning embeddings (128 dimensions)
    EMBEDDING_FEATURES = [str(i) for i in range(128)]

    # Combine all feature groups
    ALL_FEATURES = (VARIABILITY_FEATURES + COLOR_FEATURES + MAGNITUDE_FEATURES +
                   PERIOD_FEATURES + PERIODOGRAM_FEATURES + QUALITY_FEATURES +
                   POSITION_FEATURES + EMBEDDING_FEATURES)

    # Filter for features present in both datasets
    train_features = set(train_df.columns)
    test_features = set(test_df.columns)
    common_features = list(train_features.intersection(test_features))
    available_features = [f for f in ALL_FEATURES if f in common_features]

    print(f"Using {len(available_features)} features from {len(ALL_FEATURES)} candidates")
    print(f"Features: {available_features}")

    return available_features


def train_xgboost_gpu(train_df, test_df, features, label_col, output_file):
    # [First part of function remains unchanged...]
    
    # Save results
    test_df_result.to_csv(output_file, index=False)
    print(f"\nSaved predictions to {output_file}")

    # Feature importance analysis
    importance = model.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    print("\nTop 20 features by importance:")
    for feature, score in sorted_importance[:20]:
        print(f"  {feature}: {score:.4f}")

    # Create enhanced visualizations
    feature_names = [x[0] for x in sorted_importance]
    importance_values = [x[1] for x in sorted_importance]

    # Define class_names outside the conditional block (needed for all visualizations)
    class_names = label_encoder.classes_
    
    # Use the enhanced feature importance plot
    plot_xgb_feature_importance(feature_names, importance_values, top_n=20)
    print("Saved feature importance visualization to xgb_feature_importance.png")

    # These visualizations don't need ground truth, just predictions
    try:
        # Plot confidence distribution
        plot_confidence_distribution(confs, preds, class_names)
        print("Saved confidence distribution plot to confidence_distribution.png")
        
        # Feature plots
        plot_period_amplitude(test_df_result, "xgb_predicted_class")
        plot_galactic_distribution(test_df_result, "xgb_predicted_class")
        plot_color_color(test_df_result, "xgb_predicted_class")
        plot_astronomical_map(test_df_result, "xgb_predicted_class")
        
        # These might need additional columns that may not exist
        plot_hr_diagram(test_df_result, "xgb_predicted_class")
        
        # Plot feature distributions by class
        selected_features = features[:12] if len(features) > 12 else features
        plot_feature_class_correlations(test_df_result, selected_features, "xgb_predicted_class")
        print("Saved feature class distributions to feature_class_distributions.png")
        
        # These might need additional arguments to work correctly
        plot_xgb_class_probability_heatmap(probs, class_names)
        plot_xgb_top2_confidence_scatter(probs, preds, class_names)
    except Exception as e:
        print(f"Error in visualization: {e}")

    # Evaluate if ground truth is available
    if label_col in test_df_result.columns:
        print("\nEvaluation on test set:")
        y_true = test_df_result[label_col]
        y_pred = test_df_result["xgb_predicted_class"]

        # Print classification report
        print(classification_report(y_true, y_pred))

        # Plot confusion matrix
        plot_classification_performance(y_true, y_pred, class_names)
        print("Saved confusion matrix to confusion_matrix.png")
        
        # This visualization needs ground truth
        plot_misclassification_analysis(y_true, y_pred, probs, class_names)
    else:
        print(f"\nGround truth column '{label_col}' not found in test data.")
        print("Skipping evaluation metrics that require ground truth.")

    return model, label_encoder, probs



# === Main Function ===
def main():
    if len(sys.argv) < 3:
        print("Usage: python xgb_classify.py <training_fits_file> <testing_fits_file> [output_file]")
        training_file = "../PRIMVS/PRIMVS_P_GAIA.fits"
        testing_file = "../PRIMVS/PRIMVS_P.fits"
        output_file = "xgb_predictions.csv"
        print("Proceeding withdefaults")
    else:
        training_file = sys.argv[1] if len(sys.argv) > 1 else "../PRIMVS/PRIMVS_P_GAIA.fits"
        testing_file = sys.argv[2] if len(sys.argv) > 2 else "../PRIMVS/PRIMVS_P.fits"
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
    model, label_encoder, probs = train_xgboost_gpu(train_df, test_df, features, label_col, output_file)

    print("\nProcessing complete!")

if __name__ == "__main__":
    main()
