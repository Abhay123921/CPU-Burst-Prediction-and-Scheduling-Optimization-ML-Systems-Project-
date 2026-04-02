import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load data
df = pd.read_csv("data.csv")

X = df.drop(columns=["target", "ProcessID"])
y = df["target"]

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print("Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))

# ================= LR =================
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_val_error = mean_absolute_error(y_val, lr_model.predict(X_val))
print("Linear regression model trained")

# ================= RF (CV) =================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n🔁 RF Cross Validation:")
for i, (tr, va) in enumerate(kf.split(X_train)):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train.iloc[tr], y_train.iloc[tr])
    preds = model.predict(X_train.iloc[va])
    print(f"Fold {i+1} MAE:", round(mean_absolute_error(y_train.iloc[va], preds), 2))

rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_val_error = mean_absolute_error(y_val, rf_model.predict(X_val))

# ================= LSTM =================
X_train_lstm = X_train.values.reshape(-1, 100, 1)
X_val_lstm = X_val.values.reshape(-1, 100, 1)
X_test_lstm = X_test.values.reshape(-1, 100, 1)

lstm_model = Sequential([
    LSTM(32, input_shape=(100,1)),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mae')

print("\n🚀 Training LSTM...")
lstm_model.fit(X_train_lstm, y_train, epochs=5, batch_size=32,
               validation_data=(X_val_lstm, y_val), verbose=1)

lstm_val_error = mean_absolute_error(y_val, lstm_model.predict(X_val_lstm).flatten())

# ================= TEST =================
lr_preds = lr_model.predict(X_test)
rf_preds = rf_model.predict(X_test)
lstm_preds = lstm_model.predict(X_test_lstm).flatten()
baseline = X_test.iloc[:, -1]

print("\nFINAL TEST RESULTS")
print("LR:", round(mean_absolute_error(y_test, lr_preds), 2))
print("RF:", round(mean_absolute_error(y_test, rf_preds), 2))
print("LSTM:", round(mean_absolute_error(y_test, lstm_preds), 2))
print("Baseline:", round(mean_absolute_error(y_test, baseline), 2))


pickle.dump(lr_model, open("lr_model.pkl", "wb"))
pickle.dump(rf_model, open("rf_model.pkl", "wb"))
lstm_model.save("lstm_model.keras")