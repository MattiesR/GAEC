import r0877229



if __name__ == "__main__":
	size = 50
	filename = f"greedy{size}random"
	solver = r0877229.r0877229(filename)
	solver.init_random_ratio = 0.5
	solver.init_greedy_ratio = 0.3
	population_size = 100
	solver.optimize(f"src/data/tour{size}.csv")