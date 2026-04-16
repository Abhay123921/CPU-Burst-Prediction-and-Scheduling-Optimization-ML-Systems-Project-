import numpy as np
import random
import pandas as pd
import math


# REPRODUCIBILITY

random.seed(42)
np.random.seed(42)


# TARGET FUNCTION (HARDER)

def compute_target(seq):

    # Short-term dependency
    short_term = 0.4 * seq[-1] + 0.3 * seq[-2]

    # Long-term dependency (breaks EMA)
    long_term = 0.2 * np.mean(seq[-20:])

    # Nonlinear interaction
    nonlinear = (
        0.3 * math.sin(seq[-3]) +
        0.2 * (seq[-5] * seq[-1]) / 50
    )

    # Regime shift
    if np.mean(seq[-10:]) > 30:
        regime = 10
    else:
        regime = -5

    # Random spike (breaks smoothing)
    spike = 0
    if random.random() < 0.1:
        spike = random.uniform(10, 25)

    # Noise
    noise = random.uniform(-5, 5)

    target = short_term + long_term + nonlinear + regime + spike + noise

    return int(max(5, min(80, target)))



# MAIN DATA GENERATION

def generate_dataset(num_processes=50000, window_size=120):

    states_list = ["cpu", "io", "interactive"]

    states = [random.choice(states_list) for _ in range(num_processes)]
    durations = [0] * num_processes
    sequences = [[] for _ in range(num_processes)]

    for t in range(window_size):

        
        # GLOBAL REGIME SHIFT
        

        if t % 40 < 20:
            system_regime = "low_load"
        else:
            system_regime = "high_load"

        
        # SYSTEM LOAD

        cpu_count = sum(1 for s in states if s == "cpu")
        io_count = sum(1 for s in states if s == "io")

        cpu_load = cpu_count / num_processes
        io_load = io_count / num_processes


        # UPDATE EACH PROCESS
        
        for i in range(num_processes):

            state = states[i]
            duration = durations[i]

    
            # STATE SWITCHING

            phase_change_prob = min(0.05 + 0.05 * duration, 0.8)

            if random.random() < phase_change_prob:

                if state == "cpu":
                    probs = {"interactive": 0.7, "io": 0.3}

                elif state == "interactive":
                    probs = {"cpu": 0.5, "io": 0.5}

                else:
                    probs = {"interactive": 0.7, "cpu": 0.3}

                state = random.choices(
                    list(probs.keys()), list(probs.values())
                )[0]

                duration = 0
            else:
                duration += 1

            
            # BASE VALUE GENERATION

            if state == "cpu":
                val = np.random.normal(35, 8)

            elif state == "io":
                val = np.random.exponential(10)

            else:
                val = np.random.normal(20, 12)

        
            # GLOBAL REGIME EFFECT

            if system_regime == "high_load":
                val *= 1.5
            else:
                val *= 0.8


            # NONLINEAR SLOWDOWN

            val = val * (1 / (1 + 2 * cpu_load + 1.5 * io_load))

    
            # RANDOM SPIKE

            if random.random() < 0.1:
                val += random.uniform(10, 30)

        
            # PERIODIC PATTERN (hidden)

            val += 5 * math.sin(2 * math.pi * t / 25)

        
            # REDUCED TEMPORAL SMOOTHNESS

            if len(sequences[i]) >= 3:
                weights = [0.5, 0.3, 0.2]
                window = sequences[i][-3:]
                avg = sum(w * x for w, x in zip(weights, window[::-1]))

                val = 0.3 * avg + 0.7 * val   # 🔥 reduced smoothing

            
            # CLAMP

            val = int(max(5, min(80, val)))

            sequences[i].append(val)
            states[i] = state
            durations[i] = duration

        # progress
        if (t + 1) % 20 == 0:
            print(f"Time step {t+1}/{window_size} completed")


    # BUILD DATASET

    data = []

    for i in range(num_processes):
        seq = sequences[i]
        target = compute_target(seq)
        data.append(seq + [target])

    columns = [f"prev{i}" for i in range(window_size, 0, -1)] + ["target"]

    df = pd.DataFrame(data, columns=columns)

    df.insert(0, "ProcessID", [f"P{i+1}" for i in range(num_processes)])

    df.to_csv("data.csv", index=False)

    print("\n✅ Dataset generated successfully!")
    print("Shape:", df.shape)



# RUN

if __name__ == "__main__":
    generate_dataset()
