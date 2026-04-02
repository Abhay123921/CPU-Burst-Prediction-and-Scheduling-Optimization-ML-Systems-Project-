# CPU Burst Prediction & Scheduling Optimization (ML System)

## 🚀 Overview
This project builds a machine learning-based system to predict CPU burst times and optimize process scheduling decisions. It combines workload simulation, real-world trace data, and multiple ML models to improve scheduling efficiency over traditional heuristics.

---

## 🧠 Problem Statement
Traditional CPU scheduling algorithms (e.g., FCFS, SJF) rely on static assumptions and fail to adapt to dynamic workloads.

👉 This project aims to:
- Predict CPU burst times using ML
- Improve scheduling decisions using predictions
- Compare performance with traditional algorithms

---

## ⚙️ Key Features

### 🔹 Workload Simulation
- Designed a realistic simulator for:
  - CPU-bound processes  
  - IO-bound processes  
  - Interactive workloads  
- Generated large-scale dataset:
  - **50K processes × 100 timesteps**

---

### 🔹 Real-World Data Processing
- Processed **Google Cluster Trace dataset (~390K samples)**
- Handled:
  - Irregular formats  
  - Nested CPU usage distributions  
- Converted raw data into structured time-series format

---

### 🔹 Machine Learning Models
Implemented and compared:
- Linear Regression  
- Random Forest  
- LSTM  

---

### 🔹 Key Insight
- Observed strong temporal smoothness in CPU usage  
- Tree-based models outperformed LSTM in short-term prediction  

👉 Important real-world insight for system design

---

### 🔹 Scheduling Optimization
- Integrated ML predictions into scheduling logic  
- Compared against heuristic methods (e.g., SJF)  
- Achieved:
  - Reduced waiting time  
  - Improved turnaround time  

---

### 🔹 End-to-End ML Pipeline
- Data generation  
- Preprocessing  
- Model training & evaluation  
- Deployment:
  - FastAPI (backend)
  - Streamlit (UI)

---

## 🛠 Tech Stack
- Python  
- Scikit-learn  
- TensorFlow / Keras  
- Pandas, NumPy  
- FastAPI  
- Streamlit  

---

## 📂 Project Structure
scripts/ → preprocessing & training
api.py → prediction API
app.py → Streamlit interface
scheduler.py → scheduling logic
train_model.py → model training


---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train_model.py
python app.py
