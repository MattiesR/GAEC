import pandas as pd
import matplotlib.pyplot as plt


def plot_convergence(filename):
    # Load the convergence data
    filename = f"{filename}.csv"
    df = pd.read_csv(filename, comment="#")

    # Extract columns
    iterations = df.iloc[:, 0]
    elapsed_time = df.iloc[:, 1]
    mean_objective = df.iloc[:, 2]
    best_objective = df.iloc[:, 3]

    # --- Print summary statistics ---
    print("=== Convergence Summary ===")
    print(f"Data loaded from       : {filename}")
    print(f"Total iterations       : {len(iterations)}")
    print(f"Total elapsed time [s] : {elapsed_time.iloc[-1]:.4f}")
    print()

    print(f"Best objective reached : {best_objective.min():.6f}")
    print(f"Final best objective   : {best_objective.iloc[-1]:.6f}")
    print(f"Initial best objective : {best_objective.iloc[0]:.6f}")
    print()

    print(f"Initial mean objective : {mean_objective.iloc[0]:.6f}")
    print(f"Final mean objective   : {mean_objective.iloc[-1]:.6f}")
    print(f"Best mean improvement  : {mean_objective.iloc[0] - mean_objective.min():.6f}")
    print()

    print(f"Average improvement per iteration: "
        f"{(best_objective.iloc[0] - best_objective.min()) / len(iterations):.6f}")
    print("============================\n")

    # --- Plotting ---
    plt.figure(figsize=(8, 5))
    plt.plot(elapsed_time, mean_objective, 'o-', label='Mean objective', linewidth=1.8, markersize=4)
    plt.plot(elapsed_time, best_objective, 's-', label='Best objective', linewidth=1.8, markersize=4)

    plt.title(f"Convergence Graph: Iters= {len(iterations)}", fontsize=13)
    plt.xlabel("Time [s]", fontsize=12)
    plt.ylabel("Objective Value", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    size = 500
    filename = f"greedy{size}random"
    plot_convergence(filename)