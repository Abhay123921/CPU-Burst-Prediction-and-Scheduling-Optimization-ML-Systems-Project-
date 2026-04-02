import pandas as pd
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "raw", "borg_traces_data.csv")

df = pd.read_csv(file_path, engine="python", on_bad_lines="skip")

# ============================
# CLEAN CPU COLUMN
# ============================

def parse_cpu(x):
    try:
        if pd.isna(x):
            return None
        
        # remove brackets
        x = str(x).replace("[", "").replace("]", "")
        
        # split by whitespace (IMPORTANT)
        values = x.split()
        
        # convert to float
        return [float(v) for v in values]
    
    except:
        return None

df["cpu_seq"] = df["cpu_usage_distribution"].apply(parse_cpu)

# drop invalid rows
df = df.dropna(subset=["cpu_seq"])

print("Valid sequences:", len(df))
print("Sample parsed:", df["cpu_seq"].iloc[0])

# save
df.to_pickle(os.path.join(BASE_DIR, "processed", "clean.pkl"))