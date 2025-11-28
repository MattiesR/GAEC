import r0877229
import optuna


def objective(trial):
	solver = r0877229.r0877229()
	solver.population_size = trial.suggest_int("pop_size", 100, 200)
	solver.mutation_rate = trial.suggest_float("mut_rate", 0.01, 0.5)
	solver.crossover_rate = trial.suggest_float("cross_rate", 0.5, 1.0)

	solver.patience = 1000
	solver.init_greedy_ratio = 0.0
	solver.init_bfs_ratio =0.0
	solver.init_dfs_ratio = 0.0
	solver.optimize("./src/data/tour250.csv")

	# After optimization, return the best objective (lower is better)
	return solver.best_objective  # or whatever your reporter tracks



if __name__ == "__main__":
	# Create a study
	study = optuna.create_study(direction="minimize")  # we want to minimize tour length

	# Run optimization
	study.optimize(objective, n_trials=50)  # try 50 sets of hyperparameters

	# Print results
	print("Best trial:")
	trial = study.best_trial
	print(f"  Value: {trial.value}")
	print("  Params: ")
	for key, value in trial.params.items():
		print(f"    {key}: {value}")
    