import pandas as pd
import numpy as np
import logging
import os
from sklearn.preprocessing import MinMaxScaler

#CONFIGURABLE
DATASET_ID = "FD001"  # <-- Change this to FD002, FD003, FD004 as needed
BASE_DIR = "CMAPSSData"
INPUT_PATH = os.path.join(BASE_DIR, f"train_{DATASET_ID}.txt")
OUTPUT_PATH = os.path.join(BASE_DIR, "transformed", f"transformed_{DATASET_ID.lower()}.csv")
LOG_PATH = os.path.join(BASE_DIR, "transformed", f"preprocessing_log_{DATASET_ID.lower()}.txt")
RUL_CAP = None  # Set to e.g., 125 if you want to clip long life cycles

#LOGGING SETUP
logging.basicConfig(
    filename=LOG_PATH,
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

#LOAD RAW DATA
def load_raw_data(path):
    cols = ['unit', 'time', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
           [f'sensor_{i}' for i in range(1, 22)]
    df = pd.read_csv(path, sep=' ', header=None)
    df.dropna(axis=1, inplace=True)
    df.columns = cols
    logging.info(f"Loaded raw data from {path} with shape: {df.shape}")
    return df

#RENAME COLUMNS
def rename_columns(df):
    sensor_rename = {
        'sensor_2': 'stage_temp_x',
        'sensor_3': 'spindle_vibration_x',
        'sensor_4': 'spindle_vibration_y',
        'sensor_7': 'coolant_pressure',
        'sensor_8': 'coolant_flow_rate',
        'sensor_11': 'airflow_alignment_error',
        'sensor_12': 'voltage_fluctuation',
        'sensor_15': 'thermal_drift_z',
        'sensor_17': 'motor_torque',
        'sensor_20': 'load_variation',
        'sensor_21': 'power_draw'
    }
    op_rename = {
        'op_setting_1': 'chamber_temp_setpoint',
        'op_setting_2': 'pressure_mode',
        'op_setting_3': 'rpm_config'
    }
    df.rename(columns={**sensor_rename, **op_rename}, inplace=True)
    logging.info("Renamed selected sensors and operational settings.")
    return df

#DROP LOW VARIANCE
def drop_low_variance(df, threshold=0.01):
    sensor_cols = [col for col in df.columns if 'sensor_' in col or any(x in col for x in ['temp', 'vibration', 'pressure', 'flow', 'drift', 'torque', 'power'])]
    low_var = [col for col in sensor_cols if df[col].std() < threshold]
    df.drop(columns=low_var, inplace=True)
    logging.info(f"Dropped low-variance sensors: {low_var}")
    return df

#NOISE INJECTION
def inject_precision_noise(df, level=0.003):
    noise_cols = [col for col in df.columns if any(x in col for x in ['vibration', 'pressure', 'flow', 'temp', 'drift'])]
    for col in noise_cols:
        df[col] += np.random.normal(loc=0, scale=level, size=len(df))
    logging.info(f"Injected Gaussian noise (σ={level}) into: {noise_cols}")
    return df

#DRIFT SIMULATION
def simulate_drift(df, rate=0.0005, drift_cols=['thermal_drift_z']):
    for col in drift_cols:
        if col in df.columns:
            drift = df.groupby('unit').cumcount() * rate
            df[col] += drift
    logging.info(f"Applied linear drift to: {drift_cols}")
    return df

#RESCALE SENSOR VALUES
def rescale_sensors(df, scale_factor=0.001):
    scale_cols = [col for col in df.columns if any(x in col for x in ['vibration', 'pressure', 'flow', 'temp', 'drift', 'torque', 'power'])]
    df[scale_cols] *= scale_factor
    logging.info(f"Rescaled sensor values by {scale_factor} for: {scale_cols}")
    return df

#PER-UNIT NORMALIZATION
def normalize_per_unit(df):
    norm_cols = [col for col in df.columns if any(x in col for x in ['vibration', 'pressure', 'flow', 'temp', 'drift', 'torque', 'power'])]
    scaler = MinMaxScaler()
    df[norm_cols] = df.groupby('unit')[norm_cols].transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).flatten())
    logging.info("Applied per-unit MinMax normalization.")
    return df

#COMPUTE RUL
def compute_rul(df, cap=None):
    max_cycles = df.groupby('unit')['time'].transform('max')
    df['RUL'] = max_cycles - df['time']
    if cap:
        df['RUL'] = df['RUL'].clip(upper=cap)
        logging.info(f"RUL capped at {cap}")
    logging.info("Computed RUL.")
    return df

#MAIN PIPELINE
def full_preprocess_pipeline():
    df = load_raw_data(INPUT_PATH)
    df = rename_columns(df)
    df = drop_low_variance(df)
    df = inject_precision_noise(df)
    df = simulate_drift(df)
    df = rescale_sensors(df)
    df = normalize_per_unit(df)
    df = compute_rul(df, cap=RUL_CAP)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"✅ Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    full_preprocess_pipeline()
    print(f"All preprocessing steps for {DATASET_ID} applied.")
