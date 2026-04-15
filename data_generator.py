import numpy as np
import random
import pandas as pd
import math


random.seed(42)
np.random.seed(42)


# TARGET FUNCTION

def compute_target(seq):
    weights = [0.25, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05]

    linear = sum(w * seq[-(i+1)] for i, w in enumerate(weights))

    nonlinear = (
        0.2 * math.sin(seq[-3]) +
        0.2 * (seq[-1] ** 1.2) / 10
    )

    momentum = 0.15 * (seq[-1] - seq[-2])
    noise = random.uniform(-2, 2)

    target = linear + nonlinear + momentum + noise
    return int(max(5, min(60, target)))



# MAIN DATA GENERATION

def generate_dataset(num_processes=50000, window_size=100):

    states_list = ["cpu", "io", "interactive"]

    # initialize
    states = [random.choice(states_list) for _ in range(num_processes)]
    durations = [0] * num_processes
    sequences = [[] for _ in range(num_processes)]

    for t in range(window_size):
        
        # 🔥 SYSTEM LOAD
        cpu_count = sum(1 for s in states if s == "cpu")
        io_count = sum(1 for s in states if s == "io")

        cpu_load = cpu_count / num_processes
        io_load = io_count / num_processes

        # UPDATE EACH PROCESS

        for i in range(num_processes):

            state = states[i]
            duration = durations[i]

            # DURATION-BASED SWITCHING

            phase_change_prob = min(0.05 + 0.05 * duration, 0.8)

            # LOAD-AWARE TRANSITIONS

            if random.random() < phase_change_prob:

                if state == "cpu":
                    if cpu_load > 0.7:
                        probs = {"interactive": 0.7, "io": 0.3}
                    else:
                        probs = {"interactive": 0.8, "io": 0.2}

                elif state == "interactive":
                    probs = {"cpu": 0.5, "io": 0.5}

                else:  # io
                    if io_load > 0.6:
                        probs = {"interactive": 0.8, "cpu": 0.2}
                    else:
                        probs = {"interactive": 0.7, "cpu": 0.3}

                choices = list(probs.keys())
                weights = list(probs.values())
                state = random.choices(choices, weights=weights)[0]

                duration = 0

            else:
                duration += 1

            # BASE VALUE GENERATION

            if state == "cpu":
                val = np.random.normal(30, 5)

            elif state == "io":
                val = np.random.exponential(8)

            else:
                val = np.random.normal(20, 10)

            # NONLINEAR SYSTEM SLOWDOWN

            val = val * (1 / (1 + 2 * cpu_load + 1.5 * io_load))

            
            # WAITING EFFECT (soft queue)

            if state == "cpu" and cpu_load > 0.8 and random.random() < 0.3:
                val = np.random.normal(5, 2)

            # TEMPORAL CORRELATION

            if len(sequences[i]) >= 3:
                weights = [0.5, 0.3, 0.2]
                window = sequences[i][-3:]
                avg = sum(w * x for w, x in zip(weights, window[::-1]))
                val = 0.6 * avg + 0.4 * val

            # CLAMP

            val = int(max(5, min(60, val)))

            # store
            sequences[i].append(val)
            states[i] = state
            durations[i] = duration

        # progress log
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




if __name__ == "__main__":
    generate_dataset()
