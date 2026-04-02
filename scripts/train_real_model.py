import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ============================
# LOAD DATA
# ============================

df = pd.read_csv("processed/real_sequences_k3.csv")

print("Dataset shape:", df.shape)

X = df.drop(columns=["target"])
X = X.iloc[:, :-1]
y = df["target"]

# ============================
# OPTIONAL: FEATURE ENGINEERING
# ============================

X["mean"] = X.mean(axis=1)
X["std"] = X.std(axis=1)

# ============================
# TRAIN / VAL / TEST SPLIT
# ============================

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print("Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))

# ============================
# LINEAR REGRESSION
# ============================

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_val_error = mean_absolute_error(y_val, lr_model.predict(X_val))
print("\nLR Val MAE:", round(lr_val_error, 4))

# ============================
# RANDOM FOREST
# ============================

rf_model = RandomForestRegressor(n_estimators=100, n_jobs=-1)
rf_model.fit(X_train, y_train)

rf_val_error = mean_absolute_error(y_val, rf_model.predict(X_val))
print("RF Val MAE:", round(rf_val_error, 4))

# ============================
# LSTM (WINDOW = 10)
# ============================

WINDOW = X.shape[1] - 2  # minus mean + std

X_train_lstm = X_train.iloc[:, :WINDOW].values.reshape(-1, WINDOW, 1)
X_val_lstm = X_val.iloc[:, :WINDOW].values.reshape(-1, WINDOW, 1)
X_test_lstm = X_test.iloc[:, :WINDOW].values.reshape(-1, WINDOW, 1)

lstm_model = Sequential([
    LSTM(32, input_shape=(WINDOW, 1)),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mae')

print("\n🚀 Training LSTM...")
lstm_model.fit(
    X_train_lstm, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_val_lstm, y_val),
    verbose=1
)

lstm_val_preds = lstm_model.predict(X_val_lstm).flatten()
lstm_val_error = mean_absolute_error(y_val, lstm_val_preds)

print("LSTM Val MAE:", round(lstm_val_error, 4))

# ============================
# TEST RESULTS
# ============================

lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)
lstm_preds = lstm_model.predict(X_test_lstm).flatten()

baseline = X_test.iloc[:, WINDOW-1]

print("\nFINAL TEST RESULTS")
print("LR:", round(mean_absolute_error(y_test, lr_preds), 4))
print("RF:", round(mean_absolute_error(y_test, rf_preds), 4))
print("LSTM:", round(mean_absolute_error(y_test, lstm_preds), 4))
print("Baseline:", round(mean_absolute_error(y_test, baseline), 4))

# ============================
# SAVE MODELS
# ============================

os.makedirs("models", exist_ok=True)

pickle.dump(lr_model, open("models/lr_real.pkl", "wb"))
pickle.dump(rf_model, open("models/rf_real.pkl", "wb"))
lstm_model.save("models/lstm_real.keras")

print("\n✅ Models saved successfully!")