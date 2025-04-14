from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from joblib import load
import shap
import io, os, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from joblib import load as joblib_load
from tcn import TCN

#Configuration
MODEL_PATH = "saved_models/fd001/rul_classifier.joblib"
SCALER_PATH = "saved_models/fd001/classifier_scaler.joblib"
FEATURES_PATH = "saved_models/fd001/classifier_features.txt"
DATA_PATH = "CMAPSSData/transformed/transformed_fd001.csv"

# === Load Artifacts ===
model = load(MODEL_PATH)
scaler = load(SCALER_PATH)
with open(FEATURES_PATH) as f:
    features = [line.strip() for line in f]
df = pd.read_csv(DATA_PATH)
tcn_feature_columns = [col for col in df.columns if col not in ['unit', 'time', 'RUL']]

#Feature Engineering
def add_rolling_features(df, window=5):
    sensors = [col for col in df.columns if any(x in col for x in ['vibration', 'pressure', 'temp', 'drift', 'flow', 'torque', 'power'])]
    for col in sensors:
        df[f"{col}_rollmean"] = df.groupby('unit')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f"{col}_rollstd"] = df.groupby('unit')[col].transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        df[f"{col}_delta"] = df.groupby('unit')[col].diff().fillna(0)
    return df

df = add_rolling_features(df)

#flask App
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if "unit_id" not in data:
        return jsonify({"error": "Missing 'unit_id' parameter"}), 400
    try:
        unit_id = int(data["unit_id"])
    except ValueError:
        return jsonify({"error": "unit_id must be an integer"}), 400

    unit_data = df[df['unit'] == unit_id].sort_values('time')
    if unit_data.empty:
        return jsonify({"error": f"No data found for unit id {unit_id}"}), 404

    #CYCLE BASED PREDICTIO
    if "cycle" in data:
        try:
            cycle = int(data["cycle"])
        except ValueError:
            return jsonify({"error": "cycle must be an integer"}), 400

        record = unit_data[unit_data["time"] == cycle]
        if record.empty:
            return jsonify({"error": f"Cycle {cycle} not found for unit {unit_id}"}), 404
        record_for_shap = record.copy()
    else:
        record = unit_data.tail(30)
        record_for_shap = record.tail(1)

    X = record[features]
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    pred_class = int(pd.Series(preds).mode()[0])
    confidence = float(np.mean(np.max(model.predict_proba(X_scaled), axis=1)))

    #SHAP on last row in scope
    explainer = shap.TreeExplainer(model)
    last_scaled = scaler.transform(record_for_shap[features])
    shap_vals = explainer.shap_values(last_scaled)
    local_shap = shap_vals[:, :, pred_class][0]

    shap.initjs()
    expl = shap.Explanation(
        values=local_shap,
        base_values=explainer.expected_value[pred_class],
        data=last_scaled[0],
        feature_names=features
    )
    plt.figure()
    shap.plots.waterfall(expl, max_display=15, show=False)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return jsonify({
        "unit_id": unit_id,
        "cycle": data.get("cycle", "last_30"),
        "predicted_class": pred_class,
        "confidence_score": confidence,
        "shap_plot": plot_base64
    })

#Load TCN artifacts
TCN_MODEL_PATH = "saved_models/fd001/tcn_model.h5"
TCN_SCALER_PATH = "saved_models/fd001/tcn_scaler.joblib"
WINDOW_SIZE = 30

tcn_model = load_model(TCN_MODEL_PATH, custom_objects={"TCN": TCN})
tcn_scaler = joblib_load(TCN_SCALER_PATH)

@app.route('/predict_rul', methods=['POST'])
def predict_rul():
    data = request.get_json()

    if "unit_id" not in data:
        return jsonify({"error": "Missing 'unit_id' parameter"}), 400
    try:
        unit_id = int(data["unit_id"])
    except ValueError:
        return jsonify({"error": "unit_id must be an integer"}), 400

    unit_df = df[df["unit"] == unit_id].sort_values("time")
    if unit_df.empty:
        return jsonify({"error": f"No data found for unit id {unit_id}"}), 404

    if "cycle" in data:
        try:
            cycle = int(data["cycle"])
        except ValueError:
            return jsonify({"error": "cycle must be an integer"}), 400
        unit_df = unit_df[unit_df["time"] <= cycle]

    if len(unit_df) < WINDOW_SIZE:
        return jsonify({"error": f"Unit {unit_id} does not have enough data points (need at least {WINDOW_SIZE})"}), 400

    #Take last WINDOW_SIZE readings and scale
    sequence = unit_df.tail(WINDOW_SIZE)[tcn_feature_columns]
    sequence_scaled = tcn_scaler.transform(sequence)
    X = sequence_scaled.reshape(1, WINDOW_SIZE, len(tcn_feature_columns))
    predicted_rul = float(tcn_model.predict(X)[0][0])

    return jsonify({
        "unit_id": unit_id,
        "cycle": data.get("cycle", f"cycle {unit_df['time'].max()}"),
        "predicted_rul": round(predicted_rul, 2)
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
