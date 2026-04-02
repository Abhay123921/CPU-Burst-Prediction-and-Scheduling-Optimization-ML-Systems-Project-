import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_pickle(os.path.join(BASE_DIR, "processed", "clean.pkl"))

all_data = []

WINDOW = 8
K = 3   # 🔥 predict 3 steps ahead (you can try 5 later)

for seq in df["cpu_seq"]:
    
    if len(seq) >= WINDOW + K:
        
        for i in range(len(seq) - WINDOW - K + 1):
            
            X = seq[i:i+WINDOW]
            y = seq[i + WINDOW + K - 1]   # 🔥 future target
            
            all_data.append(X + [y])

print("Total samples:", len(all_data))

columns = [f"prev{i}" for i in range(WINDOW, 0, -1)] + ["target"]

final_df = pd.DataFrame(all_data, columns=columns)

# ============================
# NORMALIZATION
# ============================

max_val = final_df.iloc[:, :-1].max().max()

final_df.iloc[:, :-1] = final_df.iloc[:, :-1] / max_val
final_df["target"] = final_df["target"] / max_val

# ============================
# SAVE
# ============================

processed_path = os.path.join(BASE_DIR, "processed")
os.makedirs(processed_path, exist_ok=True)

file_path = os.path.join(processed_path, "real_sequences_k3.csv")

final_df.to_csv(file_path, index=False)

print("Saved at:", file_path)