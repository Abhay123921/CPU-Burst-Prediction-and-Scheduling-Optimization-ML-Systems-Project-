# CPU Burst Prediction & Scheduling Optimization (ML Systems Project)

## Overview

This project develops an intelligent system for **CPU burst prediction and scheduling optimization** using machine learning. It combines **synthetic workload simulation** and **real-world Google cluster traces** to evaluate when ML models outperform classical OS heuristics.

---

## Problem Statement

Traditional CPU scheduling algorithms (e.g., FCFS, SJF) rely on simple heuristics such as last-burst estimation or exponential averaging. These approaches struggle under:

* Non-stationary workloads
* Nonlinear burst patterns
* Regime shifts (changing system load)

This project aims to:

* Predict CPU burst times using ML models
* Compare against **OS-standard heuristics (Exponential Averaging)**
* Evaluate **system-level impact** (waiting time, turnaround time)

---

## System Design

```text
Workload → Feature Pipeline → ML Models → Scheduler → System Metrics
```

---

## Workload Simulation (Advanced)

Designed a realistic simulator modeling:

* CPU-bound, IO-bound, interactive processes
* Load-aware state transitions
* Temporal dependencies

### Enhanced Complexity (NEW)

To make the problem non-trivial and ML-relevant:

* Long-term dependencies (breaks simple heuristics)
* Nonlinear interactions
* Regime switching (high-load vs low-load phases)
* Random spikes and noise
* Hidden periodic patterns

📊 Dataset:

* **50K processes × 120 timesteps (~6M datapoints)**

---

## Real-World Data Pipeline

Processed **Google Cluster Trace dataset (~390K samples)**:

* Parsed irregular string-encoded CPU distributions
* Converted into structured time-series sequences
* Built supervised datasets using sliding window

### Key Transformation:

```text
[CPU sequence] → (past window → future prediction)
```

---

## Machine Learning Models

Implemented and compared:

* **Linear Regression** → baseline linear model
* **Random Forest** → nonlinear pattern learning
* **LSTM** → sequential deep learning model

### Key Insight

* CPU bursts exhibit strong short-term locality
* Tree-based models outperform LSTM for short-horizon prediction
* LSTM becomes useful only under complex/non-stationary patterns

---

## Baseline Heuristics (OS-Inspired)

Compared ML models against:

### 1. Last Burst (Naive)

```text
next ≈ last value
```

### 2. Exponential Averaging (OS Standard)

```text
τₙ₊₁ = α·tₙ + (1-α)·τₙ
```

✔ Smooths noise
✔ Adapts to recent trends

---

## Critical Insight (IMPORTANT)

* On simple workloads → **Exponential Averaging outperforms ML**
* On complex workloads → **ML outperforms heuristics**

This demonstrates:

> “ML is not always necessary—its benefit depends on workload complexity.”

---

## Scheduling Evaluation

Simulated scheduling using predicted burst times:

Measured:

* Waiting Time
* Turnaround Time
* Prediction Latency

Compared:

* Heuristic baseline (EMA)
* ML-based predictions

---

## End-to-End ML Pipeline

* Data generation (synthetic simulator)
* Real data preprocessing
* Feature engineering
* Model training & validation
* System-level evaluation (scheduler)
* Deployment-ready APIs

---

## Key Learnings

* Short-term smoothness → simple heuristics sufficient
* Complex/non-stationary workloads → ML necessary
* Prediction accuracy ≠ system performance (must evaluate scheduler impact)
* Trade-off between **latency vs accuracy** is critical in systems

---

## Tech Stack

* Python
* Scikit-learn
* TensorFlow / Keras
* Pandas, NumPy
* FastAPI
* Streamlit

---

## Project Structure

```text
OS/
├── data_generator.py        # synthetic workload simulation
├── train_model.py          # synthetic model training
├── scheduler.py            # scheduling evaluation
├── scripts/
│   ├── pre_process.py      # real data cleaning
│   ├── sequence_builder.py # sequence creation
│   ├── train_real_model.py # real-data training
├── processed/              # processed datasets
├── models/                 # saved models
├── api.py                  # FastAPI backend
├── app.py                  # Streamlit UI
```

---

## How to Run

### Synthetic Pipeline

```bash
python data_generator.py
python train_model.py
python scheduler.py
```

### Real Data Pipeline

```bash
cd scripts
python pre_process.py
python sequence_builder.py
python train_real_model.py
```

---

## Future Improvements

* True SJF scheduling using ML-based ordering
* Hybrid scheduler (EMA + ML switching)
* Online learning for adaptive workloads
* Distributed scheduling simulation

---

## 🏁 Conclusion

This project goes beyond standard ML tasks by integrating:

* Machine learning
* Operating systems concepts
* System-level evaluation

It highlights a key systems insight:

> **“The effectiveness of ML depends on the structure and complexity of the underlying workload.”**

---
