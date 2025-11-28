import pandas as pd
import matplotlib.pyplot as plt

# Load the convergence data
filename = "greedy500random.csv"
df = pd.read_csv(filename, comment="#")

# Extract columns
iterations = df.iloc[:, 0]
elapsed_time = df.iloc[:, 1]
mean_objective = df.iloc[:, 2]
best_objective = df.iloc[:, 3]

# --- Plotting ---
plt.figure(figsize=(8, 5))
plt.plot(iterations, mean_objective, 'o-', label='Mean objective', linewidth=1.8, markersize=4)
plt.plot(iterations, best_objective, 's-', label='Best objective', linewidth=1.8, markersize=4)

plt.title("Convergence Graph", fontsize=13)
plt.xlabel("Iteration", fontsize=12)
plt.ylabel("Objective Value", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Optional: log scale if objective spans orders of magnitude
# plt.yscale('log')
plt.show()