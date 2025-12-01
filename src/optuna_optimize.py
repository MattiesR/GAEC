import argparse
import csv
import r0877229
import optuna


# ------------------------------
# Objective function
# ------------------------------
def objective(trial, size):
    solver = r0877229.r0877229()

    # Population params
    solver.population_size = trial.suggest_int("pop_size", 50, 300)
    
    # Variation params
    solver.crossover_rate = trial.suggest_float("cross_rate", 0.6, 1.0)
    solver.mutation_patience = trial.suggest_float("mut_patience", 20, 200)
    solver.mut_high = trial.suggest_float("mut_high", 0.2, 0.8)
    solver.mut_low = trial.suggest_float("mut_low", 0.0, 0.4)
    
    # Stopping criteria
    solver.max_iterations = int(1e5)
    solver.patience = trial.suggest_int("patience", 100, 300)
    
    # Initialization
    solver.init_greedy_ratio = trial.suggest_float("init_greedy_ratio", 0.0, 1.0)
    solver.init_random_ratio = 1 - solver.init_greedy_ratio

    # Selection
    solver.k_tournament = trial.suggest_int("k_tourn", 1, 10)
    solver.elitism_ratio = trial.suggest_float("elitism", 0.0, 0.10)
    
    # Mutation scheme ratios
    solver.swap_ratio = trial.suggest_float("swap", 0.10, 0.60)
    solver.inversion_ratio = trial.suggest_float("invers", 0.10, 1 - solver.swap_ratio - 0.05)
    solver.scramble_ratio = 1 - (solver.swap_ratio + solver.inversion_ratio)

    # Local search
    solver.local_search_probability = trial.suggest_float("LSO_prob", 0.0, 0.4)
    solver.K_lso = trial.suggest_int("K-nearest", 5, 50)
    solver.max_improvement_lso = trial.suggest_int("max_improv", 1, 20)

    # Run solver
    solver.optimize(f"./src/data/tour{size}.csv")
    return solver.best_objective



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize GA for TSP of N cities using optuna")
    parser.add_argument("N", type=int, help="Number of cities")
    args = parser.parse_args()

    size = args.N
    n_trials_total = 2e3  # Maximum amount of trials

    # --- Persistent study ---
    study_name = f"gaec_study_tour{size}"
    storage_name = f"sqlite:///optuna/{study_name}.db"

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        storage=storage_name,
        load_if_exists=True  # continue existing study if it exists
    )

    objective_fun = lambda trial: objective(trial, size)

    # How many trials have already been done?
    n_existing = len(study.trials)
    n_remaining = max(0, n_trials_total - n_existing)

    print(f"Existing trials: {n_existing}, remaining trials to run: {n_remaining}")

    if n_remaining > 0:
        objective_fun = lambda trial: objective(trial, size)
        study.optimize(objective_fun, n_trials=n_remaining)
    else:
        print("Study already has the requested total number of trials.")


    # Save best trial separately
    with open(f"optuna/optuna_best_trial_{size}.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["parameter", "value"])
        for k, v in study.best_trial.params.items():
            writer.writerow([k, v])
        writer.writerow(["objective", study.best_trial.value])

    print("Finished. Results written to:")
    print(f" - optuna/optuna_best_trial_{size}.csv (best parameters)")