import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump

# === CONFIG ===
INPUT_PATH = "CMAPSSData/transformed/transformed_fd001.csv"
RANDOM_SEED = 42
ROLLING_WINDOW = 5
SAVE_DIR = "saved_models/fd001"
os.makedirs(SAVE_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(INPUT_PATH)

# === FEATURE ENGINEERING ===
def add_rolling_features(df, window=5):
    sensors = [col for col in df.columns if any(x in col for x in ['vibration', 'pressure', 'temp', 'drift', 'flow', 'torque', 'power'])]
    for col in sensors:
        df[f"{col}_rollmean"] = df.groupby('unit')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f"{col}_rollstd"] = df.groupby('unit')[col].transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f"{col}_delta"] = df.groupby('unit')[col].diff().fillna(0)
    return df

df = add_rolling_features(df, window=ROLLING_WINDOW)

# === TRAIN/TEST SPLIT ===
features = [col for col in df.columns if col not in ['unit', 'time', 'RUL']]
X = df[features]
y = df['RUL']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# === SCALING ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === TRAIN BASELINE MODELS ===
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_SEED),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=RANDOM_SEED)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    results[name] = {"MAE": mae, "RMSE": rmse}
    print(f"\n{name} Results:")
    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# === SAVE MODELS AND METADATA ===
for name, model in models.items():
    dump(model, os.path.join(SAVE_DIR, f"{name.lower()}_model.joblib"))
dump(models[min(results, key=lambda x: results[x]['RMSE'])], os.path.join(SAVE_DIR, "best_model.joblib"))
dump(scaler, os.path.join(SAVE_DIR, "scaler.joblib"))
with open(os.path.join(SAVE_DIR, "features.txt"), 'w') as f:
    for feat in features:
        f.write(f"{feat}\n")
print(f"✅ All models, scaler, and features saved in {SAVE_DIR}")

# === PLOT PREDICTIONS FOR BEST MODEL ===
best_model = min(results, key=lambda x: results[x]['RMSE'])
best_preds = models[best_model].predict(X_test_scaled)
plt.figure(figsize=(10, 4))
plt.plot(y_test.values[:200], label='True RUL')
plt.plot(best_preds[:200], label=f'{best_model} Prediction')
plt.title(f'RUL Prediction (Best: {best_model})')
plt.xlabel('Sample Index')
plt.ylabel('RUL')
plt.legend()
plt.tight_layout()
plt.savefig('fd001_rul_prediction_plot.png')
plt.show()
