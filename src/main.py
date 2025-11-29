import r0877229
import pandas as pd
import matplotlib.pyplot as plt
import plots


if __name__ == "__main__":
	size = 1000
	filename = f"greedy{size}random"
	solver = r0877229.r0877229(filename)
	solver.init_random_ratio = 0.0
	solver.init_greedy_ratio = 1.0
	solver.init_bfs_ratio = 0.0
	solver.init_dfs_ratio = 0.0
	# solver.mutation_rate = 0.03
	solver.population_size = 80
	solver.max_iterations = 10e3
	solver.patience = 1e3
	solver.optimize(f"src/data/tour{size}.csv")
	plots.plot_convergence(filename)