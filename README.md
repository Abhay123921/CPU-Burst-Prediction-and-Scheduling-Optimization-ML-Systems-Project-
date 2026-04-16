# CPU Burst Prediction & ML-Driven Scheduling Optimization

## Overview

This project builds an **end-to-end ML-driven CPU scheduling system** that predicts CPU burst times and evaluates their impact on scheduling performance.

It combines:

* **Synthetic workload simulation** (controlled experimentation)
* **Real-world Google cluster traces (~390K samples)** (practical validation)

The system demonstrates **when machine learning improves over classical OS heuristics—and when it does not**.

---

## Problem Statement

Traditional CPU scheduling algorithms (e.g., FCFS, SJF) rely on simple heuristics such as:

* Last-burst estimation
* Exponential averaging (EMA)

These approaches break down under:

* Non-stationary workloads
* Nonlinear burst patterns
* Dynamic system load (regime shifts)

### Objectives:

* Predict CPU burst times using ML models
* Compare against **OS-standard heuristics (EMA)**
* Evaluate **system-level performance** (not just prediction accuracy)

---

## System Architecture

```text
Workload → Feature Pipeline → ML Models → Scheduler → System Metrics
```

---

## ⚙️ Workload Simulation (Synthetic)

A realistic simulator was designed to model:

* CPU-bound, IO-bound, and interactive processes
* Load-aware state transitions
* Temporal dependencies

### Enhanced Complexity

To create a non-trivial ML problem:

* Long-term dependencies (breaks EMA)
* Nonlinear feature interactions
* Regime switching (high-load vs low-load)
* Random spikes and noise
* Hidden periodic patterns

📊 **Dataset Scale**:

* ~50,000 processes × 120 timesteps
* ≈ **6 million datapoints**

---

## Real-World Data Pipeline

Processed **Google Cluster Trace dataset (~390K samples)**:

* Parsed irregular string-encoded CPU usage distributions
* Converted into structured time-series sequences
* Built supervised datasets via sliding window

### Transformation:

```text
[CPU sequence] → (past window → future burst)
```

---

## Machine Learning Models

Implemented and compared:

* **Linear Regression** → baseline
* **Random Forest** → nonlinear modeling
* **LSTM** → sequence modeling

### Key Findings

* CPU bursts show strong **short-term locality**
* **Random Forest outperforms LSTM** for short-horizon prediction
* LSTM is only beneficial under **complex/non-stationary patterns**

---

## Baseline Heuristics (OS-Inspired)

### 1. Last Burst (Naive)

```text
next ≈ last value
```

### 2. Exponential Averaging (EMA)

```text
τₙ₊₁ = α·tₙ + (1-α)·τₙ
```

✔ Smooths noise
✔ Adapts to recent trends

---

## Critical Insight

* **Simple workloads → EMA outperforms ML**
* **Complex workloads → ML outperforms heuristics**

> ML is beneficial **only when workload complexity justifies it**

---

## 🖥️ Scheduling Evaluation

Implemented **ML-driven SJF scheduling**:

* Predictions used to **order processes**
* Execution simulated using **actual burst times**

### 📏 Metrics Evaluated

* Waiting Time
* Turnaround Time
* Prediction Latency

### Result

* **~18–20% reduction in waiting time** over EMA baseline (Random Forest)
* Demonstrates **real system-level gains from ML**

---

## End-to-End ML Pipeline

* Synthetic data generation
* Real-world data preprocessing
* Feature engineering
* Model training & validation
* Scheduling simulation
* Deployment-ready APIs

---

## Deployment

Deployed as an interactive ML system:

* **FastAPI** → backend inference API
* **Streamlit** → interactive dashboard

### Features:

* Real-time CPU burst prediction
* Model comparison (LR / RF / LSTM)
* Scheduling simulation (ML vs EMA)
* Latency monitoring

---

## Key Learnings

* Prediction accuracy ≠ system performance
* Short-term smoothness → heuristics sufficient
* Non-stationarity → ML becomes useful
* Trade-off between **latency vs accuracy** is critical
* ML should be used **only when it adds value**

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
├── scheduler.py            # ML-based SJF scheduling
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

## Future Work

* Hybrid scheduler (EMA + ML switching)
* Preemptive scheduling (SRJF)
* Online learning for adaptive workloads
* Distributed scheduling simulation

---

## Conclusion

This project bridges **Machine Learning + Operating Systems + Systems Design**.

> **“The effectiveness of ML depends on workload structure—simple heuristics often suffice, but ML becomes powerful under complexity and non-stationarity.”**
