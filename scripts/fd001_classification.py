import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.model_selection import StratifiedShuffleSplit

#CONFIG
INPUT_PATH = "CMAPSSData/transformed/transformed_fd001.csv"
RANDOM_SEED = 42
ROLLING_WINDOW = 5
SAVE_DIR = "saved_models/fd001"
os.makedirs(SAVE_DIR, exist_ok=True)

#LOAD DATA
df = pd.read_csv(INPUT_PATH)

#FEATURE ENGINEERING
def add_rolling_features(df, window=5):
    sensors = [col for col in df.columns if any(x in col for x in ['vibration', 'pressure', 'temp', 'drift', 'flow', 'torque', 'power'])]
    for col in sensors:
        df[f"{col}_rollmean"] = df.groupby('unit')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f"{col}_rollstd"] = df.groupby('unit')[col].transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f"{col}_delta"] = df.groupby('unit')[col].diff().fillna(0)
    return df

df = add_rolling_features(df, window=ROLLING_WINDOW)

#CLASSIFICATION LABELING
def classify_rul(rul):
    if rul > 100:
        return 0  # GREEN
    elif 50 < rul <= 100:
        return 1  # YELLOW
    else:
        return 2  # RED

df['RUL_Class'] = df['RUL'].apply(classify_rul)

#TRAIN/TEST SPLIT
features = [col for col in df.columns if col not in ['unit', 'time', 'RUL', 'RUL_Class']]
X = df[features]
y = df['RUL_Class']
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_SEED)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

#SCALING
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#CLASSIFICATION MODEL
clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, class_weight='balanced')
clf.fit(X_train_scaled, y_train)
preds = clf.predict(X_test_scaled)

#EVALUATION
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=['Green (>100)', 'Yellow (50-100)', 'Red (<=50)']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, preds))

#SAVE CLASSIFICATION MODEL
dump(clf, os.path.join(SAVE_DIR, "rul_classifier.joblib"))
dump(scaler, os.path.join(SAVE_DIR, "classifier_scaler.joblib"))
print(f"Classification model saved to {SAVE_DIR}")

# Save the features used for the classifier
with open(os.path.join(SAVE_DIR, "classifier_features.txt"), 'w') as f:
    for feat in features:
        f.write(f"{feat}\n")
