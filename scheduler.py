import pandas as pd
import numpy as np
import pickle
import time
from tensorflow.keras.models import load_model


# LOAD DATA

df = pd.read_csv("data.csv")

X = df.drop(columns=["ProcessID", "target"])
y = df["target"]

train_size = int(0.7 * len(df))
val_size = int(0.15 * len(df))

X_test = X.iloc[train_size + val_size:]
y_test = y.iloc[train_size + val_size:]



# LOAD MODELS

lr_model = pickle.load(open("lr_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))
lstm_model = load_model("lstm_model.keras", compile=False)



# HELPER FUNCTIONS

def prepare_lstm_input(x):
    return x.values.reshape(1, x.shape[0], 1)


def exponential_avg(history, alpha=0.6):
    tau = history.iloc[0]
    for t in history.iloc[1:]:
        tau = alpha * t + (1 - alpha) * tau
    return tau


def simulate(process_list):
    waiting = 0
    turnaround = 0
    current_time = 0

    for p in process_list:
        waiting += current_time
        current_time += p["actual"]
        turnaround += current_time

    return waiting, turnaround



# MAIN SCHEDULER

def run_scheduler(num_processes=50, model_type="rf"):

    print(f"\n🚀 Running scheduler using: {model_type.upper()}")

    total_latency = 0
    processes = []

    
    # STEP 1: PREDICTION

    for i in range(num_processes):

        history = X_test.iloc[i]
        actual = y_test.iloc[i]

        start = time.time()

        if model_type == "lr":
            pred = lr_model.predict(history.to_frame().T)[0]

        elif model_type == "rf":
            pred = rf_model.predict(history.to_frame().T)[0]

        elif model_type == "lstm":
            pred = lstm_model.predict(
                prepare_lstm_input(history), verbose=0
            )[0][0]

        else:
            raise ValueError("Invalid model type")

        latency = time.time() - start
        total_latency += latency

        pred = max(1, pred)

        baseline = exponential_avg(history, alpha=0.6)

        processes.append({
            "id": i,
            "pred": pred,
            "actual": actual,
            "baseline": baseline
        })

    
    # STEP 2: SORT (SJF)

    ml_sorted = sorted(processes, key=lambda x: x["pred"])
    baseline_sorted = sorted(processes, key=lambda x: x["baseline"])

    
    # STEP 3: SIMULATE EXECUTION

    ml_waiting, ml_turnaround = simulate(ml_sorted)
    baseline_waiting, baseline_turnaround = simulate(baseline_sorted)


    # RESULTS

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")

    print("\nML Waiting Time:", round(ml_waiting, 2))
    print("Baseline Waiting Time:", round(baseline_waiting, 2))

    print("\nML Turnaround Time:", round(ml_turnaround, 2))
    print("Baseline Turnaround Time:", round(baseline_turnaround, 2))

    print("\nAvg Latency per Prediction:",
          round(total_latency / num_processes, 6), "sec")

    improvement = ((baseline_waiting - ml_waiting) / baseline_waiting) * 100

    print("\n🔥 Improvement over baseline:",
          round(improvement, 2), "%")



# RUN

if __name__ == "__main__":

    for model in ["lr", "rf", "lstm"]:
        run_scheduler(num_processes=50, model_type=model)
