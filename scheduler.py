import pandas as pd
import numpy as np
import pickle
import time
from tensorflow.keras.models import load_model

# LOAD DATA

df = pd.read_csv("data.csv")

# drop ProcessID
X = df.drop(columns=["ProcessID", "target"])
y = df["target"]

# split same as training
train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))

X_test = X.iloc[train_size + val_size:]
y_test = y.iloc[train_size + val_size:]


# LOAD MODELS

lr_model = pickle.load(open("lr_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
lstm_model = load_model("lstm_model.keras", compile=False)


# HELPER: LSTM INPUT

def prepare_lstm_input(x):
    return x.values.reshape(1, x.shape[0], 1)


# SCHEDULER FUNCTION

def run_scheduler(num_processes=50, model_type="rf"):

    print(f"\n🚀 Running scheduler using: {model_type.upper()}")

    waiting_time = 0
    turnaround_time = 0

    baseline_waiting = 0
    baseline_turnaround = 0

    total_latency = 0

    for i in range(num_processes):

        history = X_test.iloc[i]
        actual = y_test.iloc[i]

        
        # ML PREDICTION

        start = time.time()

        if model_type == "lr":
            pred = lr_model.predict([history])[0]

        elif model_type == "rf":
            pred = rf_model.predict([history])[0]

        elif model_type == "lstm":
            pred = lstm_model.predict(
                prepare_lstm_input(history), verbose=0
            )[0][0]

        else:
            raise ValueError("Invalid model type")

        latency = time.time() - start
        total_latency += latency

        pred = max(1, pred)  # safety

        
        # BASELINE (LAST BURST)

        baseline = history.iloc[-1]

    
        # WAITING TIME (CUMULATIVE)

        waiting_time += pred
        turnaround_time += waiting_time + actual

        baseline_waiting += baseline
        baseline_turnaround += baseline_waiting + actual

        
        # PRINT PER PROCESS

        print(f"\nProcess {i+1}")
        print("Actual:", actual)
        print("Prediction:", round(pred, 2))
        print("Baseline:", baseline)
        print("ML Error:", round(abs(pred - actual), 2))
        print("Baseline Error:", abs(baseline - actual))
        print("Latency:", round(latency, 4), "sec")

    
    # FINAL RESULTS

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")

    print("\nML Waiting Time:", round(waiting_time, 2))
    print("Baseline Waiting Time:", round(baseline_waiting, 2))

    print("\nML Turnaround Time:", round(turnaround_time, 2))
    print("Baseline Turnaround Time:", round(baseline_turnaround, 2))

    print("\nAvg Latency per Prediction:",
          round(total_latency / num_processes, 6), "sec")

    improvement = ((baseline_waiting - waiting_time) / baseline_waiting) * 100

    print("\n🔥 Improvement over baseline:",
          round(improvement, 2), "%")



# RUN ALL MODELS

if __name__ == "__main__":

    for model in ["lr", "rf", "lstm"]:
        run_scheduler(num_processes=50, model_type=model)
