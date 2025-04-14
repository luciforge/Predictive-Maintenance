import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from joblib import load
import os
from shap import Explanation

#CONFIG
DATA_PATH = "CMAPSSData/transformed/transformed_fd001.csv"
MODEL_PATH = "saved_models/fd001/rul_classifier.joblib"
SCALER_PATH = "saved_models/fd001/classifier_scaler.joblib"
FEATURES_PATH = "saved_models/fd001/classifier_features.txt"

#LOAD ARTIFACTS
model = load(MODEL_PATH)
scaler = load(SCALER_PATH)
with open(FEATURES_PATH, 'r') as f:
    features = [line.strip() for line in f.readlines()]

# === LOAD DATA ===
df = pd.read_csv(DATA_PATH)

def classify_rul(rul):
    if rul > 100:
        return 0  # Green
    elif 50 < rul <= 100:
        return 1  # Yellow
    else:
        return 2  # Red

df['RUL_Class'] = df['RUL'].apply(classify_rul)

def add_rolling_features(df, window=5):
    sensors = [col for col in df.columns if any(x in col for x in ['vibration', 'pressure', 'temp', 'drift', 'flow', 'torque', 'power'])]
    for col in sensors:
        df[f"{col}_rollmean"] = df.groupby('unit')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f"{col}_rollstd"] = df.groupby('unit')[col].transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f"{col}_delta"] = df.groupby('unit')[col].diff().fillna(0)
    return df

df = add_rolling_features(df, window=5)
X = df[features]
y = df['RUL_Class']
X_scaled = pd.DataFrame(scaler.transform(X), columns=features)
print("Loaded features count:", len(features))
print("X_scaled shape:", X_scaled.shape)

#SHAP ANALYSIS (TREE BASED)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled)  # Expected shape: (n_samples, n_features, n_classes)
print("SHAP values shape:", shap_values.shape)  # This should print: (20631, 42, 3)

#GLOBAL SUMMARY PLOTS FOR ALL CLASSES
for class_idx, class_name in enumerate(['Green (>100)', 'Yellow (50-100)', 'Red (<=50)']):
    # Extract SHAP values for the given class across all samples:
    class_shap_values = shap_values[:, :, class_idx]
    # Generate summary plot:
    shap.summary_plot(class_shap_values, X_scaled, feature_names=features, show=False)
    plt.tight_layout()
    safe_class_name = (class_name.replace(' ', '_')
                                     .replace('(', '')
                                     .replace(')', '')
                                     .replace('<=', 'lte')
                                     .replace('<', 'lt')
                                     .replace('>', 'gt')
                                     .replace('-', '_'))
    filename = f"fd001_shap_summary_class{class_idx}_{safe_class_name}.png"
    plt.savefig(filename)
    plt.close()
    print(f" SHAP summary plot saved for {class_name}")

#FORCE PLOTS FOR SELECTED SAMPLES (ALL CLASSES)
sample_ids = [100, 500, 1000]  # arbitrary interesting samples
shap.initjs()  # Initialize JS once outside the loop
for idx in sample_ids:
    for class_idx, class_name in enumerate(['Green', 'Yellow', 'Red']):
        plt.figure()
        # Extract the shap values for sample idx and class class_idx.
        # sample_shap has shape (42,)
        sample_shap = shap_values[idx, :, class_idx]
        # Create an Explanation object with the expected value (from the explainer),
        # the sample's data (from X_scaled), and the list of feature names.
        expl = Explanation(
            values=sample_shap,
            base_values=explainer.expected_value[class_idx],
            data=X_scaled.iloc[idx],
            feature_names=features
        )
        # Now create the waterfall plot using the Explanation object.
        shap.plots.waterfall(expl, max_display=15, show=False)
        plt.title(f"SHAP Force Plot for Sample {idx} (Class: {class_name})")
        plt.tight_layout()
        fname = f"shap_force_sample_{idx}_class{class_idx}_{class_name}.png"
        plt.savefig(fname)
        plt.close()
        print(f"SHAP force plot saved: {fname}")

