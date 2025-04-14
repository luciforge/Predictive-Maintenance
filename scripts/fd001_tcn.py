import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tcn import TCN
from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping
from joblib import dump

#CONFIGURATION
DATA_PATH = "CMAPSSData/transformed/transformed_fd001.csv"
WINDOW_SIZE = 30   # Number of cycles to use as sequence input (adjust as needed)
BATCH_SIZE = 32
EPOCHS = 50         # You can adjust as necessary
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

#LOAD DATA
df = pd.read_csv(DATA_PATH)

# For regression, we use all columns except 'unit', 'time', and 'RUL' as features
features = [col for col in df.columns if col not in ['unit', 'time', 'RUL']]
target = 'RUL'

#Creating Sequences
#For a TCN, we need to create sequences (windows) per unit.
#We group the data by engine ("unit") and slide a window over each engine’s lifecycle.
def create_sequences(df, window_size, feature_columns, target_column):
    X, y = [], []
    # Group by unit
    for unit, group in df.groupby('unit'):
        group = group.sort_values('time')
        data = group[feature_columns].values
        targets = group[target_column].values
        #create sliding windows; each sequence's target is the last cycle's RUL in that window.
        if len(data) >= window_size:
            for i in range(len(data) - window_size + 1):
                seq_x = data[i: i + window_size]
                seq_y = targets[i + window_size - 1]  #predict the RUL at the end of the window
                X.append(seq_x)
                y.append(seq_y)
    return np.array(X), np.array(y)

# Create sequences
X_seq, y_seq = create_sequences(df, WINDOW_SIZE, features, target)
print("Sequences shape:", X_seq.shape)  # Expected: (n_samples, WINDOW_SIZE, n_features)
print("Target shape:", y_seq.shape)

#SCALING
#We reshape X_seq to 2D for scaling then reshape back.
n_samples, win_size, n_features = X_seq.shape
X_reshaped = X_seq.reshape(-1, n_features)  # shape = (n_samples*win_size, n_features)

scaler = StandardScaler()
X_scaled_2d = scaler.fit_transform(X_reshaped)
dump(scaler, "saved_models/fd001/tcn_scaler.joblib")
X_scaled = X_scaled_2d.reshape(n_samples, win_size, n_features)

#Split into train and test sets (using a random 80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_seq, test_size=0.2, random_state=RANDOM_SEED)

print("Train set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

#BUILD THE TCN MODEL
#We build a model with an input shape of (WINDOW_SIZE, n_features)
input_layer = Input(shape=(WINDOW_SIZE, n_features))
#The TCN layer automatically applies causal convolutions and dilation
tcn_layer = TCN(nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8], dropout_rate=0.05, return_sequences=False)(input_layer)
output_layer = Dense(1)(tcn_layer)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='mae')

model.summary()

#TRAIN THE MODEL
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
history = model.fit(X_train, y_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_split=0.2,
                    callbacks=[early_stop],
                    verbose=1)
model.save("saved_models/fd001/tcn_model.h5")

#EVALUATE MODEL
test_loss = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE: {test_loss:.2f}")

#plot the training/validation loss curve
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MAE")
plt.legend()
plt.tight_layout()
plt.savefig("tcn_loss_plot.png")
plt.show()

#PREDICTION EXAMPLE
#Plot predicted vs true RUL for the first 200 samples of the test set
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 4))
plt.plot(y_test[:200], label='True RUL')
plt.plot(y_pred[:200], label='Predicted RUL', linestyle='--')
plt.title("TCN RUL Prediction (First 200 Test Samples)")
plt.xlabel("Sample Index")
plt.ylabel("RUL")
plt.legend()
plt.tight_layout()
plt.savefig("tcn_rul_prediction_plot.png")
plt.show()
