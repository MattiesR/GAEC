import r0877229


if __name__ == "__main__":
	size = 500
	filename = f"greedy{size}random"
	solver = r0877229.r0877229(filename)
	# solver.init_random_ratio = 0.0
	# solver.init_greedy_ratio = 1.0
	# solver.init_bfs_ratio = 0.0
	# solver.init_dfs_ratio = 0.0
	solver.mutation_rate = 0.5
	# solver.population_size = 150
	solver.patience = 100
	solver.optimize(f"src/data/tour{size}.csv")