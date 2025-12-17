import Reporter
import numpy as np
from numba import njit, prange

class r0877229:
	# -------------------
	# Hyperparameters
	# -------------------
	""" Population params """
	population_size = 200

	""" Variation params """
	crossover_rate = 0.55
	mutation_rate = 0.1
	mutation_patience = 50
	mutation_increase = 0.05	
	mut_high = 0.3
	mut_low = 0.1


	""" Initialization params """
	init_random_ratio = 0.0
	init_greedy_ratio = 1.0
	noise_init_greedy = 0.01	
	init_bfs_ratio = 0.0
	init_dfs_ratio = 0.0
	init_vectorized_random_ratio = 0.0
	""" Selection params """
	k_tournament = 3 
	elitism_ratio = 0.05	# Default as 5%
	
	""" Variation params """
	
	# Ratios Mutations (weights of mutation schemes)
	swap_ratio = 0.55
	inversion_ratio = 0.35	
	scramble_ratio = 0.10 	# Occasional low probability

	""" Local search params """
	local_search_probability = 0.15
	lso_max = 0.45      # start aggressive
	lso_min = 0.25    # keep a little LS forever

	max_improv_two_opt = 20
	max_improv_three_opt = 5
	max_improv_seg_swap = 20
    

	min_seg = 3
	max_seg = 15
      

	""" Island diversity"""
	island_diversity_init = 0.2
	island_diversity_rules = 0.3
	

	""" Fitness sharing """
	fitness_sharing_treshold = 0.35
	sigma = 0.6
	alpha = 1.2
	# -------------------
	# Objective function
	# -------------------
	best_objective = np.inf
	mean_objective = np.inf
	

	def __init__(self, filename=None):
		# Global-only hyperparameters (instance attributes) 
		""" Stopping criterea params"""
		self.max_iterations = 1e9
		self.patience = 3e9


		""" Diversity promotion """
		self.num_islands = 6				 # 8
		self.migration_interval = 200	 # 100

		""" Diversity per Island """
		self.island_diversity_init = 0.2  # Diversity in the initialization of islands
		self.island_diversity_rules = 0.2 # Different rules at different islands

		if filename is None:
			filename = self.__class__.__name__
		self.reporter = Reporter.Reporter(filename)

	
	# -------------------
	# Main optimization loop
	# -------------------
	def optimize(self, filename):
		distance_matrix = np.genfromtxt(
			filename,
			delimiter=",",
			missing_values="Inf",
			filling_values=np.inf
		)

		# Initialize islands
		islands_size = self.population_size // self.num_islands
		islands = [Island(islands_size,i) for i in range(self.num_islands)]
	
		for i, island in enumerate(islands):
			island.initialize(distance_matrix)
			island.apply_island_diversity(i, islands_size, self.island_diversity_rules)
		self.print_islands_rules(islands)

		iteration = 0
		no_improvement = 0

		while iteration < self.max_iterations:
			""" REMOVE BEFORE HANDING IN: """
			if distance_matrix.shape[0] == 15:
				self.max_iterations = 100
			# --- Genetic operations per island ---
			# --- For each island ---
			for island in islands:
				# Generate next generation
				island.next_generation(distance_matrix)
				
				# Find best individual in this island
				best_idx = np.argmin(island.fitness)
				best_obj = island.fitness[best_idx]
				
				# --- Update per-island no_improvement ---
				if best_obj < island.best_objective:
					# Improvement found → reset counter
					island.best_objective = best_obj
					island.no_improvement = 0
				else:
					# No improvement → increment counter
					island.no_improvement += 1
				
				# Adaptive mutation can use the updated counter
				island.adaptive_mutation()

			# --- Migration ---
			if iteration % self.migration_interval == 0:
				self.migrate(islands, distance_matrix)

			# --- Best per island ---
			best_per_island = [isl.best() for isl in islands]
			populations = [isl.population for isl in islands]
			hamming_divs_per_island = all_islands_diversity_numba(populations)
			for (i,hamming_div) in enumerate(hamming_divs_per_island):
				islands[i].update_effective_crossover_rate(hamming_div)

			# --- Global best ---
			all_fitness = np.concatenate([isl.fitness for isl in islands])
			all_population = np.concatenate([isl.population for isl in islands])
			best_idx = np.argmin(all_fitness)
			best_solution = all_population[best_idx]
			best_objective = all_fitness[best_idx]
			mean_objective = np.mean(all_fitness)


			# --- Reporting ---
			time_left = self.reporter.report(mean_objective, best_objective, best_solution)
			print("Best per island:")
			for idx, ((_, obj), hamming) in enumerate(zip(best_per_island, hamming_divs_per_island)):
				print(f"  Island {idx}: best objective = {obj:.4f}, hamming = {hamming:.4f}")

			# Stopping criteria
			if time_left < 0:
				print("Ran out of time")
				break

			if no_improvement >= self.patience:
				print("Out of patience")
				print("Nuclear bomb comming soon")
				break
			if abs((best_objective - mean_objective)) <= 1e-5 and iteration > 100:
				print("Mean converged to best!")
				break
                  
			if best_objective < self.best_objective:
				no_improvement = 0
				self.mutation_rate = self.mut_low

			iteration += 1
			no_improvement += 1

			# Updating best objectives
			self.best_objective = best_objective
			self.mean_objective = mean_objective

			# Adaptively decrease local search 
			# # self.local_search_probability = max(self.lso_min,self.lso_max * (time_left / 300.0))	
			# p = max(0.0, min(1.0, time_left / 300.0))
			# self.local_search_probability = max(self.lso_min, self.lso_max * p * p)
			# for island in islands:
			# 	island.local_search_probability = self.local_search_probability

			print(f"Iteration: {iteration}, best = {best_objective}, mean= {mean_objective}")
		return 0



	# -------------------
	# GA Methods
	# -------------------
	# """Initialization algorithms"""
	# def initialize_population(self, num_cities, pop_size, distance_matrix=None):
	# 	"""
	# 	Initialize the population using multiple strategies.
	# 	Strategies and ratios are defined as class attributes:
	# 		self.init_methods = [
	# 			("random", self.init_random, self.init_random_ratio),
	# 			("greedy", self.init_greedy, self.init_greedy_ratio),
	# 			("bfs", self.init_graph_bfs, self.init_bfs_ratio),
	# 			("dfs", self.init_graph_dfs, self.init_dfs_ratio),
	# 		]
	# 	"""

	# 	# Build the list of (method, ratio) dynamically
	# 	methods = [
	# 		(self.init_random, self.init_random_ratio),
	# 		(self.init_greedy, self.init_greedy_ratio),
	# 		(self.init_graph_bfs, self.init_bfs_ratio),
	# 		(self.init_graph_dfs, self.init_dfs_ratio),
	# 		(self.init_vectorized_random, self.init_vectorized_random_ratio)
	# 	]
	# 	# Compute number of individuals per method
	# 	counts = [int(pop_size * ratio) for _, ratio in methods]

	# 	# Fix rounding to make total exactly pop_size
	# 	remaining = pop_size - sum(counts)
	# 	if remaining != 0:
	# 		counts[0] += remaining  # Add the difference to the first method (random)

	# 	""" Print statements"""
	# 	print("------------------------------")
	# 	print(f"Initialized population of {pop_size} individuals.")
	# 	method_names = ["Random", "Greedy", "BFS", "DFS", "random_feasible"]
	# 	for method, count in zip(method_names, counts):
	# 		print(f"{method}: {count}")
	# 	print("------------------------------")
		
	# 	"""	-------------- """
	# 	# Allocate population array
	# 	population = np.zeros((pop_size, num_cities), dtype=np.int32)

	# 	start_idx = 0
	# 	for (method, _), count in zip(methods, counts):
	# 		if count > 0:
	# 			population[start_idx:start_idx+count] = method(
	# 				distance_matrix if method != self.init_random else num_cities,	# Construction due to init_random taking other arguments
	# 				count
	# 			)
	# 			start_idx += count

		# return population

	def initialize_population(self, pop_size, distance_matrix):
		"""
		Initialize the population using greedy initialization for all individuals,
		optionally adding noise to diversify solutions.
		"""
		print("------------------------------")
		print(f"Initialized population of {pop_size} individuals using Greedy + noise.")
		print("------------------------------")
		
		# Directly call the Numba greedy initializer
		population = init_greedy_numba(distance_matrix, pop_size, noise_scale=self.noise_init_greedy)
		# population = local_search_population_2opt(population, distance_matrix, 10) # 1000 gives worse results hmmm weird
		return population

	# Random
	def init_random(self, num_cities, pop_size):
		pop = np.zeros((pop_size, num_cities), dtype=np.int32)
		for i in range(pop_size):
			pop[i] = np.random.permutation(num_cities)
		return pop

	# Greedy 
	# def init_greedy(self, distance_matrix, pop_size):
	# 	num_cities = distance_matrix.shape[0]
	# 	population = np.zeros((pop_size, num_cities), dtype=np.int32)

	# 	for k in range(pop_size):
	# 		current = np.random.randint(0, num_cities)
	# 		visited = [current]
	# 		unvisited = set(range(num_cities))
	# 		unvisited.remove(current)

	# 		while unvisited:
	# 			next_city = min(unvisited, key=lambda j: distance_matrix[current, j])
	# 			visited.append(next_city)
	# 			unvisited.remove(next_city)
	# 			current = next_city

	# 		population[k] = np.array(visited, dtype=np.int32)

	# 	return population
	
	# Greedy with optional noise
	def init_greedy(self, distance_matrix, pop_size, noise_scale=0.01):
			N = distance_matrix.shape[0]
			# --- Call the Numba function here ---
			return init_greedy_numba(distance_matrix, pop_size, N, noise_scale)



	def init_vectorized_random(self, distance_matrix, pop_size):
		n = distance_matrix.shape[0]
		pop = np.zeros((pop_size, n), dtype=np.int32)

		# --- 1. Vectorized random permutations ---
		for k in range(pop_size):
			pop[k] = np.random.permutation(n)

		# Convert ∞ to a large number for faster vector ops
		INF = np.inf
		dm = distance_matrix

		# --- 2. Repair infeasible edges (vectorized per-individual) ---
		for k in range(pop_size):
			tour = pop[k]

			for i in range(n - 1):
				u = tour[i]
				v = tour[i + 1]

				if dm[u, v] == INF:
					# find all feasible next nodes
					remaining = tour[i+1:]
					feasible_mask = (dm[u, remaining] != INF)

					if not np.any(feasible_mask):
						# fallback: choose closest feasible city (vectorized)
						feasible = np.where(dm[u] != INF)[0]
						v_new = feasible[np.argmin(dm[u, feasible])]
					else:
						# pick a random feasible city
						feasible = remaining[feasible_mask]
						v_new = np.random.choice(feasible)

					# swap positions so next city is v_new
					idx = np.where(tour == v_new)[0][0]
					tour[i+1], tour[idx] = tour[idx], tour[i+1]

			pop[k] = tour

		return pop
	# --- Graph-aware Randomized DFS ---
	def init_graph_dfs(self, distance_matrix, pop_size):
		num_cities = distance_matrix.shape[0]
		population = np.zeros((pop_size, num_cities), dtype=np.int32)
		raise NotImplemented

	# --- Graph-aware BFS ---
	def init_graph_bfs(self, distance_matrix, pop_size):
		num_cities = distance_matrix.shape[0]
		population = np.zeros((pop_size, num_cities), dtype=np.int32)

		for k in range(pop_size):
			start = np.random.randint(0, num_cities)
			visited = [False] * num_cities
			path = []

			queue = [start]
			while queue:
				node = queue.pop(0)
				if not visited[node]:
					visited[node] = True
					path.append(node)

					# Neighbors: nodes with finite distance
					neighbors = [j for j in range(num_cities)
								if distance_matrix[node, j] != np.inf and not visited[j]]

					np.random.shuffle(neighbors)  # Randomize BFS traversal
					queue.extend(neighbors)

			population[k] = np.array(path, dtype=np.int32)
		return population


	""" Evalulation of the population """
	def evaluate_population(self, population, distance_matrix):
		return evaluate_population_numba(population, distance_matrix)

	""" next generation """
	def next_generation(self, population, fitness, distance_matrix):
		"""
		(λ, µ) + elitism GA generation step.
		Parents cannot survive except for explicit elites.
		"""
		num_individuals = len(population)
		new_pop = np.zeros_like(population)

		# === 1) Diversity check ===
		global_hamming_diversity = global_hamming_diversity_numba(population)
		use_sharing = global_hamming_diversity < self.fitness_sharing_treshold
		# use_sharing = True
		if use_sharing:
			selection_fitness = fitness_sharing_numba(population, fitness, sigma=self.sigma, alpha=self.alpha)
		else:
			selection_fitness = fitness

		# === 2) ELITISM ===
		elitism = max(1,int(self.population_size * self.elitism_ratio))
		# Copy top elites directly from parents
		new_pop[:elitism] = elitism_core_numba(population, fitness, elitism)

		# Apply local search on elites
		for i in range(elitism):
			if np.random.rand() < self.local_search_probability:
				new_pop[i] = self.apply_local_search(new_pop[i], distance_matrix, global_hamming_diversity)

		# === 3) Offspring creation ===
		for i in range(elitism, num_individuals):
			parent1, parent2 = tournament_selection_numba(
				population, selection_fitness, self.k_tournament, 2
			)
			child = self.crossover(parent1, parent2)
			child = self.mutate(child)

			# Adaptive local search
			child = self.apply_local_search(child, distance_matrix, global_hamming_diversity)
			new_pop[i] = child

		# === 4) Evaluation ===
		new_fitness = evaluate_population_numba(new_pop, distance_matrix)

		# === 5) Normalization (optional) ===
		normalize_population_numba(new_pop)

		# === 6) Elimination (λ, µ) style ===
		# Only consider offspring (excluding elites) for survival
		# Elites are already in place
		offspring = new_pop[elitism:]
		offspring_fitness = new_fitness[elitism:]
		
		# Select top (num_individuals - elitism) offspring
		top_idx = np.argsort(offspring_fitness)[:num_individuals - elitism]
		new_pop[elitism:] = offspring[top_idx]
		new_fitness[elitism:] = offspring_fitness[top_idx]
		return new_pop, new_fitness
      
	def next_generation_old(self, population, fitness, distance_matrix):
		num_individuals = len(population)
		new_pop = np.zeros_like(population)


		# === Diversity check ===
		global_hamming_diversity = global_hamming_diversity_numba(population)

		use_sharing = global_hamming_diversity < self.fitness_sharing_treshold

		if use_sharing:
			selection_fitness = fitness_sharing_numba(population, fitness, sigma=self.sigma, alpha=self.alpha)
		else:
			selection_fitness = fitness
                  
		# === 1) ELITISM (Python) ===
		elitism = int(self.population_size * self.elitism_ratio)
		new_pop[:elitism] = elitism_core_numba(population, fitness, elitism)


		# === Apply local search on elites ===
		# For example, 2-opt or 3-opt
		# for i in range(elitism):
		# 	if np.random.rand() < self.local_search_probability:  # adaptive probability
		# 		# Choose which LS: 2-opt or 3-opt
		# 		new_pop[i] = two_opt_fast(new_pop[i], distance_matrix, max_improve=10)
			
		# === 2-4) Offspring creation loop ===
		for i in range(elitism, num_individuals):
			parent1, parent2 = tournament_selection_numba(
				population, selection_fitness, self.k_tournament, 2
			)
			child = self.crossover(parent1, parent2)
			child = self.mutate(child)

			# Apply adaptive local search
			child = self.apply_local_search(child, distance_matrix, global_hamming_diversity)
			new_pop[i] = child

                  

		# === 5) Evaluation phase (Numba, via Python wrapper) ===
		offspring_fitness = evaluate_population_numba(new_pop, distance_matrix)


		normalize_population_numba(new_pop) # In place normalization
		# === 6) Elimination phase (Numba) ===
		new_pop, new_fitness = elimination_numba(
			population,
			new_pop,
			fitness,
			offspring_fitness,
			num_individuals
		)
		# After elimination
		# if global_hamming_diversity < 0.05:
		# 	print("Nuke dropped")
		# 	num_nuke = max(1, int(0.1 * num_individuals))  # 10% worst
		# 	worst_idx = np.argsort(new_fitness)[-num_nuke:]  # indices of worst
		# 	for idx in worst_idx:
		# 		new_pop[idx] = init_greedy_numba(distance_matrix, 1, noise_scale=0.1)[0]
		# 		new_fitness[idx] = evaluate_individual_numba(new_pop[idx], distance_matrix)

		return new_pop, new_fitness


	""" Selection process """
	""" k-tournament selection (vectorized, faster for large populations) """

	def select_parents(self, population, fitness):
			"""
			Wrapper for tournament selection, calling the Numba core.
			"""
			parent1, parent2 = tournament_selection_numba(
					population, 
					fitness, 
					self.k_tournament, 
					2
					)
			return parent1, parent2



	""" Variation steps """
	def crossover(self, parent1, parent2):
		if np.random.rand() < self.crossover_rate_eff:
			if np.random.rand() < 0.15:
				return ordered_crossover(parent1, parent2)
			else:
				return epx_crossover(parent1, parent2)
			# return erx_fast(parent1,parent2)
		return parent1.copy()
	def update_effective_crossover_rate(self, diversity):
		"""
		diversity ∈ [0,1]
		high diversity  -> high crossover
		low diversity   -> reduced crossover
		"""

		self.crossover_rate_eff = max(
			self.crossover_rate * diversity**0.25,
			0.1
		)
	
	def edge_recombination(self,parent1, parent2):
		"""
		Edge Recombination Crossover (ERX).
		parent1, parent2: sequences (numpy arrays or lists) of city ids.
		returns a numpy array child of dtype int32 with same length.
		"""
		# convert parents to plain lists of Python ints
		p1 = list(map(int, parent1))
		p2 = list(map(int, parent2))
		size = len(p1)

		# Build adjacency map: city -> set(neighbors)
		adj = {}
		for p in (p1, p2):
			for i, city in enumerate(p):
				if city not in adj:
					adj[city] = set()
				left = p[i-1]            # wrap-around
				right = p[(i+1) % size]
				adj[city].add(left)
				adj[city].add(right)

		# Child construction
		child = []
		used = set()

		# Start from a randomly chosen city (could choose p1[0] or random)
		current = int(np.random.choice(p1))  # random start from parent1
		while len(child) < size:
			child.append(current)
			used.add(current)

			# Remove current from adjacency lists
			for nbrs in adj.values():
				if current in nbrs:
					nbrs.discard(current)

			# If all cities are used, break
			if len(child) >= size:
				break

			# Candidate neighbors (remaining neighbors of current) sorted by their adjacency size
			remaining_neighbors = [n for n in adj[current] if n not in used] if current in adj else []

			if remaining_neighbors:
				# choose the neighbor with fewest neighbors (degree). Break ties randomly.
				min_deg = None
				candidates = []
				for n in remaining_neighbors:
					deg = len([x for x in adj.get(n, set()) if x not in used])
					if (min_deg is None) or (deg < min_deg):
						min_deg = deg
						candidates = [n]
					elif deg == min_deg:
						candidates.append(n)
				current = int(np.random.choice(candidates))
			else:
				# no neighbors left -> pick a random unused city
				unused = [c for c in p1 if c not in used]
				if not unused:
					# fallback: include any city not used (shouldn't really happen)
					unused = [c for c in adj.keys() if c not in used]
				current = int(np.random.choice(unused))

		return np.array(child, dtype=np.int32)

	def mutate(self, individual):
		if np.random.rand() < self.mutation_rate:
			U = np.random.rand()
			if U < self.swap_ratio:
				return swap_mutation(individual)
			elif U < self.swap_ratio + self.inversion_ratio:
				return inversion_mutation(individual)
			else:
				return scramble_mutation(individual)
		return individual
	

	""" Default settings for hyperparameters """
	def set_mutation(self,type):
		if type == "swap":
			self.swap_ratio = 1.0
			self.inversion_ratio = 0.0
			self.scramble_ratio = 0.0
		if type == "inversion":
			self.swap_ratio = 0.0
			self.inversion_ratio = 1.0
			self.scramble_ratio = 0.0
		if type == "scramble":
			self.swap_ratio = 0.0
			self.inversion_ratio = 0.0
			self.scramble_ratio = 1.0
		assert self.swap_ratio + self.inversion_ratio + self.scramble_ratio == 1.0
            
	def apply_local_search(self, individual, distance_matrix, global_hamming_diversity):
		"""
		High diversity → more local search, mostly light (2-opt)
		Low diversity → less LS, but stronger if applied
		"""

		# # --- LS probability: increases with diversity
		# ls_prob = max(
		# 	self.local_search_probability * global_hamming_diversity,
		# 	0.15
		# )
		# if np.random.rand() >= ls_prob:
		# 	return individual
		if np.random.rand() >= self.local_search_probability:
			return individual

		# # --- LS type probabilities
		# # High diversity → 2-opt dominates
		# two_opt_prob   = 0.6 + 0.3 * global_hamming_diversity
		# seg_swap_prob  = 0.25
		# three_opt_prob = 1.0 - two_opt_prob - seg_swap_prob

		# probs = np.array([two_opt_prob, seg_swap_prob, three_opt_prob])
		# probs = np.clip(probs, 0.05, 1.0)
		# probs /= probs.sum()

		# choice = np.random.choice([0, 1, 2], p=probs)

		# if choice == 0:
		# 	return two_opt_fast(
		# 		individual, distance_matrix, self.max_improv_two_opt
		# 	)

		# elif choice == 1:
		# 	N = distance_matrix.shape[0]
		# 	segment_length = np.random.randint(
		# 		self.min_seg, min(self.max_seg, N)
		# 	)
		# 	return segment_swap_delta_safe(
		# 		individual, distance_matrix,
		# 		max_improvement=self.max_improv_seg_swap,
		# 		segment_length=segment_length
		# 	)

		# else:
		# 	return three_opt_fast(
		# 		individual, distance_matrix, self.max_improv_three_opt
		# 	)
            
		else:
			if np.random.uniform() < 0.10:
				return three_opt_fast(individual, distance_matrix,self.max_improv_two_opt)
			else:
				return or_opt_fast(individual, distance_matrix, 10, 4)


	def migrate(self, islands, distance_matrix, migrants_per_island=1):
		"""
		Ring migration: best individuals from each island move to the next island.
		Only recompute fitness for swapped individuals.
		"""
		num_islands = len(islands)
		
		# get best individual(s) per island
		best_individuals = [isl.population[np.argmin(isl.fitness)].copy() for isl in islands]

		for i in range(num_islands):
			next_island = (i + 1) % num_islands
			
			for _ in range(migrants_per_island):
				# find worst individual in the next island
				worst_idx = np.argmax(islands[next_island].fitness)
				
				# replace worst individual with the best from current island
				islands[next_island].population[worst_idx] = best_individuals[i].copy()
				
				# recompute fitness for the swapped individual only
				islands[next_island].fitness[worst_idx] = islands[next_island].evaluate_population(
					islands[next_island].population[worst_idx:worst_idx+1], distance_matrix
				)[0]

	def print_islands_rules(self, islands):
		"""
		Print all island hyperparameters in a table.
		"""
		headers = [
		"Island", "Pop", "MutRate", "CrossRate",
		"Swap", "Inv", "Scramble",
		"Greedy", "Random", "BFS", "DFS", "VectRand",
		"LocSearch", "MutPat", "MutHigh", "MutLow"
		]

		# Print header
		print(" | ".join(f"{h:>9}" for h in headers))
		print("-" * 140)

		for isl in islands:
			row = [
				getattr(isl, "idx", "?"),
				isl.population_size,
				f"{isl.mutation_rate:.3f}",
				f"{isl.crossover_rate:.3f}",
				f"{isl.swap_ratio:.3f}",
				f"{isl.inversion_ratio:.3f}",
				f"{isl.scramble_ratio:.3f}",
				f"{isl.init_greedy_ratio:.3f}",
				f"{isl.init_random_ratio:.3f}",
				f"{isl.init_bfs_ratio:.3f}",
				f"{isl.init_dfs_ratio:.3f}",
				f"{isl.init_vectorized_random_ratio:.3f}",
				f"{isl.local_search_probability:.3f}",
				isl.mutation_patience,
				f"{isl.mut_high:.3f}",
				f"{isl.mut_low:.3f}"
			]
			print(" | ".join(f"{str(r):>9}" for r in row))



class Island(r0877229):
	def __init__(self, population_size, idx=None):
		""" Island class """
		super().__init__()
		self.population_size = population_size
		self.idx = idx
		self.population = None
		self.fitness = None
		self.indiv_rules = None
		self.no_improvement = 0
		self.best_objective = np.inf
		self.crossover_rate_eff = self.crossover_rate

	def initialize(self, distance_matrix):
		self.population = self.initialize_population(self.population_size, distance_matrix)
		normalize_population_numba(self.population) # In place normalization

		self.fitness = evaluate_population_numba(self.population, distance_matrix)


	def next_generation(self, distance_matrix):
		self.population, self.fitness =  super().next_generation(self.population, self.fitness, distance_matrix)

	def adaptive_mutation(self):
		current_best = min(self.fitness)
		if current_best < self.best_objective:
			# Improvement found
			self.best_objective = current_best
			self.no_improvement = 0
			self.mutation_rate = self.mut_low
		else:
			self.no_improvement += 1
			if self.no_improvement >= self.mutation_patience:
				self.mutation_rate = self.mut_high
				# print(f"Island {self.idx}: Mutation rate increased to {self.mutation_rate}")

	def best(self):
		idx = np.argmin(self.fitness)
		return self.population[idx], self.fitness[idx]

	def apply_island_diversity(self, island_idx, num_islands, diversity_scale=0.2):
		"""
		Scale hyperparameters per island using linear + random factor.
		Ensures ratios sum to 1 for initialization and mutation operators.
		"""
		# -------------------
		# Factor for diversity
		# -------------------
		base_factor = island_idx / max(1, num_islands - 1)
		random_offset = np.random.uniform(-diversity_scale, diversity_scale)
		factor = np.clip(base_factor + random_offset, 0, 1)

		# -------------------
		# Variation parameters
		# -------------------
		self.mutation_rate = self.mut_low + factor * (self.mut_high - self.mut_low)
		self.crossover_rate = self.crossover_rate * (0.8 + 0.4*factor)

		# -------------------
		# Initialization ratios
		# -------------------
		self.init_greedy_ratio = max(0, self.init_greedy_ratio - factor*0.5)
		self.init_random_ratio = max(0, self.init_random_ratio + factor*0.5)

		init_ratios = np.array([
			self.init_greedy_ratio,
			self.init_random_ratio,
			self.init_bfs_ratio,
			self.init_dfs_ratio,
			self.init_vectorized_random_ratio
		])
		init_ratios /= init_ratios.sum()  # normalize
		(
			self.init_greedy_ratio,
			self.init_random_ratio,
			self.init_bfs_ratio,
			self.init_dfs_ratio,
			self.init_vectorized_random_ratio
		) = init_ratios
		
		
		# -------------------
		# Selection params
		# -------------------
		self.k_tournament = max(1, int(self.k_tournament * (1 + 0.2 * diversity_scale)))
		self.elitism_ratio = min(0.5, self.elitism_ratio * (1 + 0.2 * diversity_scale))
		
		# -------------------
		# Mutation operator ratios
		# -------------------
		swap = max(0, self.swap_ratio * (1 - 0.2*factor))
		inversion = max(0, self.inversion_ratio * (1 + 0.1*factor))
		scramble = max(0, self.scramble_ratio)  # keep as-is or perturb slightly

		mut_ratios = np.array([swap, inversion, scramble])
		mut_ratios /= mut_ratios.sum()  # normalize
		self.swap_ratio, self.inversion_ratio, self.scramble_ratio = mut_ratios

		# -------------------
		# Local search probability
		# -------------------
		self.local_search_probability *= (0.8 + 0.4*factor)
		# self.max_improvement_lso = max(1, int(self.max_improvement_lso * (1 + 0.2 * diversity_scale)))




def edge_set(tour):
    n = len(tour)
    return set(tuple(sorted((tour[i], tour[(i+1)%n]))) for i in range(n))

def island_distance(pop1, pop2):
    # Average edge-based distance between all pairs (one from each island)
    n1, n2 = len(pop1), len(pop2)
    dist_sum = 0
    for t1 in pop1:
        edges1 = edge_set(t1)
        for t2 in pop2:
            edges2 = edge_set(t2)
            shared = len(edges1 & edges2)
            dist_sum += 1 - shared/len(edges1)
    return dist_sum / (n1 * n2)




# -------------------
# Numba-accelerated functions
# -------------------
@njit(parallel=True)
def evaluate_population_numba(population, distance_matrix):
	n = population.shape[0]
	fitness = np.zeros(n)
	for i in prange(n):
		tour = population[i]
		total = 0.0
		for j in range(len(tour)):
			from_city = tour[j]
			to_city = tour[(j + 1) % len(tour)]
			total += distance_matrix[from_city, to_city]
		fitness[i] = total
	return fitness

@njit(cache=True)
def evaluate_individual_numba(route, distance_matrix):
    """
    Computes the total distance of a single TSP tour.

    Parameters
    ----------
    route : 1D np.ndarray
        Array of city indices (0-based) representing the tour.
    distance_matrix : 2D np.ndarray
        Symmetric distance matrix between cities.

    Returns
    -------
    float
        Total tour distance.
    """
    N = len(route)
    total = 0.0
    for i in range(N):
        a = route[i]
        b = route[(i + 1) % N]  # wrap around to form a cycle
        total += distance_matrix[a, b]
    return total

@njit(cache=True)
def swap_mutation(individual):
    """Swap two random positions in the individual."""
    a, b = np.random.randint(0, len(individual), 2)
    individual[a], individual[b] = individual[b], individual[a]
    return individual

@njit(cache=True)
def inversion_mutation(individual):
    """Invert a random segment of the individual."""
    size = len(individual)
    a, b = np.random.randint(0, size, 2)
    if a > b:
        a, b = b, a
    # reverse the segment in place
    while a < b:
        individual[a], individual[b] = individual[b], individual[a]
        a += 1
        b -= 1
    return individual

@njit(cache=True)
def scramble_mutation(individual):
    """Scramble a random segment of the individual."""
    size = len(individual)
    a, b = np.random.randint(0, size, 2)
    if a > b:
        a, b = b, a
    segment = individual[a:b+1].copy()
    np.random.shuffle(segment)
    individual[a:b+1] = segment
    return individual

@njit
def ordered_crossover(parent1, parent2):
	size = len(parent1)
	a, b = sorted(np.random.choice(size, 2, replace=False))
	child = -np.ones(size, dtype=np.int32)
	child[a:b+1] = parent1[a:b+1]
	pointer = 0
	for gene in parent2:
		if gene not in child:
			while child[pointer] != -1:
				pointer += 1
			child[pointer] = gene
	return child


@njit(cache=True)
def two_opt_fast(route, distance_matrix, 
                 max_improve=10,        # max number of improving swaps
                 candidate_list=None):  # optional list of nearest neighbors
    """
    Implements the 2-opt heuristic using the best-improvement strategy.
    Assumes route is a 0-based numpy array of city indices.
    dist is a 2D numpy array (distance matrix).
    """
    N = len(route)
    improved = True
    improve_count = 0

    while improved and improve_count < max_improve:
        improved = False
        
        for i in range(N - 1):
            # Determine candidate js
            if candidate_list is None:
                # FIX: Use np.arange to ensure 'js' is a NumPy array.
                js = np.arange(i + 2, N, dtype=np.int32) # standard full loop
            else:
                # route[i] must be used to index into candidate_list
                # since the candidate list is indexed by city index, not position
                js = candidate_list[route[i]]  # only nearest neighbors
            
            for j in js:
                if j <= i + 1 or j >= N:  # skip invalid indices
                    continue
                
                # wrap-around edges
                a = route[i - 1] if i > 0 else route[N - 1]
                b = route[i]
                c = route[j]
                d = route[(j + 1) % N]
                
                # compute delta
                delta = distance_matrix[a, c] + distance_matrix[b, d] - distance_matrix[a, b] - distance_matrix[c, d]
                
                if delta < 0:
                    # perform 2-opt swap
                    route[i:j+1] = route[i:j+1][::-1]
                    improved = True
                    improve_count += 1
                    break  # first improvement strategy
            if improved:
                break  # restart outer loop after first improvement
    return route


@njit(cache=True, parallel=True)
def local_search_population_2opt(population, distance_matrix, max_improve):
    pop_size = population.shape[0]

    for i in prange(pop_size):
        population[i] = two_opt_fast(
            population[i],
            distance_matrix,
            max_improve
        )

    return population

@njit(cache=True)
def three_opt_fast(route, distance_matrix, max_improve=5):
    N = len(route)
    improve_count = 0
    improved = True

    while improved and improve_count < max_improve:
        improved = False

        for i in range(1, N - 4):
            a = route[i - 1]
            b = route[i]

            for j in range(i + 2, N - 2):
                c = route[j - 1]
                d = route[j]

                for k in range(j + 2, N):
                    e = route[k - 1]
                    f = route[k % N]

                    # original cost
                    old = (
                        distance_matrix[a, b]
                        + distance_matrix[c, d]
                        + distance_matrix[e, f]
                    )

                    # --- CASE 1: reverse (i:j)
                    new1 = (
                        distance_matrix[a, c]
                        + distance_matrix[b, d]
                        + distance_matrix[e, f]
                    )

                    if new1 < old:
                        route[i:j] = route[i:j][::-1]
                        improved = True
                        improve_count += 1
                        break

                    # --- CASE 2: reverse (j:k)
                    new2 = (
                        distance_matrix[a, b]
                        + distance_matrix[c, e]
                        + distance_matrix[d, f]
                    )

                    if new2 < old:
                        route[j:k] = route[j:k][::-1]
                        improved = True
                        improve_count += 1
                        break

                    # --- CASE 3: reverse (i:k)
                    new3 = (
                        distance_matrix[a, c]
                        + distance_matrix[e, b]
                        + distance_matrix[d, f]
                    )

                    if new3 < old:
                        route[i:k] = route[i:k][::-1]
                        improved = True
                        improve_count += 1
                        break

                if improved:
                    break
            if improved:
                break

    return route



@njit(cache=True)
def or_opt_fast(route, distance_matrix, max_improve=5, max_seg_len=3):
    N = len(route)
    tmp = np.empty(N, dtype=route.dtype)

    improve_count = 0
    improved = True

    while improved and improve_count < max_improve:
        improved = False

        for seg_len in range(1, max_seg_len + 1):
            for i in range(1, N - seg_len):
                a = route[i - 1]
                b = route[i]
                c = route[i + seg_len - 1]
                d = route[i + seg_len]

                removed = (
                    distance_matrix[a, b]
                    + distance_matrix[c, d]
                )

                for k in range(N - 1):

                    # illegal insertion positions
                    if k >= i - 1 and k < i + seg_len:
                        continue

                    e = route[k]
                    f = route[k + 1]

                    old = removed + distance_matrix[e, f]
                    new = (
                        distance_matrix[a, d]
                        + distance_matrix[e, b]
                        + distance_matrix[c, f]
                    )

                    if new < old:
                        pos = 0

                        if k < i - 1:
                            # ---- insert BEFORE segment ----
                            # prefix [0 .. k]
                            for t in range(0, k + 1):
                                tmp[pos] = route[t]
                                pos += 1

                            # segment
                            for t in range(seg_len):
                                tmp[pos] = route[i + t]
                                pos += 1

                            # middle
                            for t in range(k + 1, i):
                                tmp[pos] = route[t]
                                pos += 1

                            # suffix
                            for t in range(i + seg_len, N):
                                tmp[pos] = route[t]
                                pos += 1

                        else:
                            # ---- insert AFTER segment ----
                            # prefix
                            for t in range(0, i):
                                tmp[pos] = route[t]
                                pos += 1

                            # middle
                            for t in range(i + seg_len, k + 1):
                                tmp[pos] = route[t]
                                pos += 1

                            # segment
                            for t in range(seg_len):
                                tmp[pos] = route[i + t]
                                pos += 1

                            # suffix
                            for t in range(k + 1, N):
                                tmp[pos] = route[t]
                                pos += 1

                        # HARD SAFETY CHECK (can remove later)
                        if pos != N:
                            raise RuntimeError("Or-opt write count mismatch")

                        for t in range(N):
                            route[t] = tmp[t]

                        improved = True
                        improve_count += 1
                        break

                if improved:
                    break
            if improved:
                break

    return route




@njit(cache=True, parallel=True)
def local_search_population_oropt(
    population,
    distance_matrix,
    max_improve,
    max_seg_len
):
    pop_size = population.shape[0]

    for i in prange(pop_size):
        population[i] = or_opt_fast(
            population[i],
            distance_matrix,
            max_improve,
            max_seg_len
        )

    return population

@njit(cache=True)
def build_adj_list(parent1, parent2):
    n = len(parent1)
    adj = -np.ones((n, 4), dtype=np.int32)
    deg = np.zeros(n, dtype=np.int32)

    def add_edge(i, j):
        for k in range(deg[i]):
            if adj[i, k] == j:
                return
        if deg[i] < 4:
            adj[i, deg[i]] = j
            deg[i] += 1

    for i in range(n):
        a = parent1[i]
        b = parent1[(i + 1) % n]
        add_edge(a, b)
        add_edge(b, a)

    for i in range(n):
        a = parent2[i]
        b = parent2[(i + 1) % n]
        add_edge(a, b)
        add_edge(b, a)

    return adj, deg

@njit(cache=True, parallel=True)
def local_search_population_3opt(population, distance_matrix, max_improve):
    pop_size = population.shape[0]

    for i in prange(pop_size):
        population[i] = three_opt_fast(
            population[i],
            distance_matrix,
            max_improve
        )
    return population


@njit(cache=True)
def remove_node(adj, deg, node):
    n = adj.shape[0]
    for i in range(n):
        d = deg[i]
        for k in range(d):
            if adj[i, k] == node:
                deg[i] -= 1
                adj[i, k] = adj[i, deg[i]]
                adj[i, deg[i]] = -1
                break

@njit(cache=True)
def erx_choose_next(current, used, adj, deg):
    best = -1
    best_deg = 999999

    for k in range(deg[current]):
        nb = adj[current, k]
        if not used[nb] and deg[nb] < best_deg:
            best_deg = deg[nb]
            best = nb

    if best != -1:
        return best

    n = len(used)
    while True:
        c = np.random.randint(0, n)
        if not used[c]:
            return c

@njit(cache=True)
def epx_choose_next(current, used, adj, deg):
    d = deg[current]
    if d > 0:
        # gather usable neighbors
        tmp = [-1, -1, -1, -1]
        count = 0
        for k in range(d):
            nb = adj[current, k]
            if not used[nb]:
                tmp[count] = nb
                count += 1
        if count > 0:
            return tmp[np.random.randint(0, count)]
    # fallback: random unused
    n = len(used)
    while True:
        c = np.random.randint(0, n)
        if not used[c]:
            return c


@njit(cache=True)
def erx_fast(parent1, parent2):
    n = len(parent1)
    adj, deg = build_adj_list(parent1, parent2)

    child = -np.ones(n, dtype=np.int32)
    used = np.zeros(n, dtype=np.bool_)

    current = parent1[0]
    child[0] = current
    used[current] = True
    remove_node(adj, deg, current)

    for pos in range(1, n):
        nxt = erx_choose_next(current, used, adj, deg)
        child[pos] = nxt
        used[nxt] = True
        remove_node(adj, deg, nxt)
        current = nxt

    return child

@njit(cache=True)
def epx_crossover(parent1, parent2):
    n = len(parent1)
    adj, deg = build_adj_list(parent1, parent2)

    child = -np.ones(n, dtype=np.int32)
    used = np.zeros(n, dtype=np.bool_)

    current = parent1[np.random.randint(0, n)]
    child[0] = current
    used[current] = True
    remove_node(adj, deg, current)

    for pos in range(1, n):
        nxt = epx_choose_next(current, used, adj, deg)
        child[pos] = nxt
        used[nxt] = True
        remove_node(adj, deg, nxt)
        current = nxt
    return child

# Initialization
@njit
def greedy_single_route(noisy_matrix, N):
    route = np.empty(N, dtype=np.int32)
    route[0] = 0
    visited = np.zeros(N, dtype=np.bool_)
    visited[0] = True
    current = 0

    for i in range(1, N):
        min_dist = np.inf
        next_city = -1
        for city in range(N):
            if not visited[city]:
                dist = noisy_matrix[current, city]
                if dist < min_dist:
                    min_dist = dist
                    next_city = city
        route[i] = next_city
        visited[next_city] = True
        current = next_city
    return route

# -------------------------------
# Main initialization function
# -------------------------------
@njit(parallel=True, cache=True)
def init_greedy_numba(distance_matrix, pop_size, noise_scale):
	N = distance_matrix.shape[0]
	population = np.zeros((pop_size, N), dtype=np.int32)

	for k in prange(pop_size):
		# Optional noise
		if noise_scale > 0.0:
			noise = np.random.uniform(1.0 - noise_scale, 1.0 + noise_scale, size=distance_matrix.shape)
			noisy_matrix = distance_matrix * noise
			noisy_matrix = np.nan_to_num(noisy_matrix, nan=np.inf)	# Makes sure no NaN numbers enter and fail the initialization
		else:
			noisy_matrix = distance_matrix

		# Construct route using the helper
		population[k] = greedy_single_route(noisy_matrix, N)
	return population

@njit(cache=True)
def elimination_numba(
    population_old, 
    population_new, 
    fitness_old, 
    fitness_new, 
    num_individuals
):
    """
    Performs the (mu + lambda) selection elimination step to choose the best 
    individuals from the combined parent and offspring populations.
    """
    
    # 1. Combine population and fitness arrays
    combined_pop = np.vstack((population_old, population_new))
    combined_fitness = np.concatenate((fitness_old, fitness_new))
    
    # 2. Select the best mu individuals using argpartition
    # np.argpartition is a highly optimized NumPy function (works great in Numba)
    # It efficiently finds the indices of the 'num_individuals' best fitness values.
    best_indices = np.argpartition(combined_fitness, num_individuals)[:num_individuals]
    
    # 3. Return the selected population and fitness arrays
    return combined_pop[best_indices], combined_fitness[best_indices]




@njit(cache=True)
def elitism_core_numba(population, fitness, elitism_count):
	"""
	Finds the indices of the 'elitism_count' best individuals 
	based on fitness (assuming lower is better).
	"""
	if elitism_count > 0:
		# np.argpartition finds the indices of the best 'elitism_count' fitness values.
		elite_idx = np.argpartition(fitness, elitism_count)[:elitism_count]
		return population[elite_idx]

	# Return an empty array if no elitism is used

	return np.empty((0, population.shape[1]), dtype=population.dtype)


@njit(cache=True)
def tournament_selection_numba(population, fitness, k_tournament, num_parents):
    """
    Performs k-tournament selection and returns the chosen parents.
    """
    pop_size = len(population)
    parents = np.empty((num_parents, population.shape[1]), dtype=population.dtype)

    for p in range(num_parents):
        # 1. Choose k random indices (Numba-compatible sampling)
        competitor_indices = np.random.choice(pop_size, size=k_tournament, replace=False)
        
        # 2. Find the winner's index (assuming lower fitness is better)
        winner_index = -1
        best_fitness = np.inf
        
        for idx in competitor_indices:
            if fitness[idx] < best_fitness:
                best_fitness = fitness[idx]
                winner_index = idx
                
        parents[p] = population[winner_index]

    # Return parent individuals, not just indices
    return parents[0], parents[1]




@njit
def segment_swap_delta_safe(route, distance_matrix, max_improvement=10, segment_length=2):
    N = route.shape[0]
    length = 0.0
    for k in range(N):
        length += distance_matrix[route[k], route[(k+1)%N]]

    improvements = 0
    attempts = 0
    max_attempts = 50_000  # prevent infinite loops

    while improvements < max_improvement and attempts < max_attempts:
        attempts += 1
        i = np.random.randint(0, N - segment_length + 1)
        j = np.random.randint(0, N - segment_length + 1)
        if abs(i - j) < segment_length:
            continue

        # Neighbors
        a = route[i-1] if i > 0 else route[N-1]
        b = route[i+segment_length] if (i+segment_length) < N else route[0]
        x = route[j-1] if j > 0 else route[N-1]
        y = route[j+segment_length] if (j+segment_length) < N else route[0]

        # Delta
        old_edges = distance_matrix[a, route[i]] + distance_matrix[route[i+segment_length-1], b] + \
                    distance_matrix[x, route[j]] + distance_matrix[route[j+segment_length-1], y]
        new_edges = distance_matrix[a, route[j]] + distance_matrix[route[j+segment_length-1], b] + \
                    distance_matrix[x, route[i]] + distance_matrix[route[i+segment_length-1], y]

        delta = new_edges - old_edges

        if delta < 0:
            # Swap safely
            temp = route[i:i+segment_length].copy()
            route[i:i+segment_length] = route[j:j+segment_length]
            route[j:j+segment_length] = temp
            length += delta
            improvements += 1

    return route


@njit(parallel=True, cache=True)
def normalize_population_numba(population):
    pop_size, N = population.shape
    tmp_pop = np.zeros_like(population)  # initialize to avoid -1

    for k in prange(pop_size):
        route = population[k]
        idx0 = 0
        for i in range(N):
            if route[i] == 0:
                idx0 = i
                break
        for i in range(N):
            tmp_pop[k, i] = route[(idx0 + i) % N]

    for k in prange(pop_size):
        for i in range(N):
            population[k, i] = tmp_pop[k, i]


@njit(parallel=True, cache=True)
def global_hamming_diversity_numba(population):
    P, N = population.shape
    total = 0.0
    count = 0

    for i in prange(P):
        for j in range(i + 1, P):
            diff = 0
            for k in range(N):
                if population[i, k] != population[j, k]:
                    diff += 1
            total += diff / N
            count += 1

    if count == 0:
        return 0.0
    return total / count


@njit(parallel=True, cache=True)
def elite_hamming_diversity_numba(population, elite):
    P, N = population.shape
    total = 0.0

    for i in prange(P):
        diff = 0
        for k in range(N):
            if population[i, k] != elite[k]:
                diff += 1
        total += diff / N

    return total / P


@njit(inline="always")
def sharing_function_numba(d, sigma, alpha):
    # assumes d < sigma
    return 1.0 - (d / sigma) ** alpha


@njit(parallel=True, cache=True)
def fitness_sharing_numba(population, fitness, sigma=0.3, alpha=1.0):
    P, N = population.shape
    shared_fitness = np.empty(P, dtype=np.float64)

    for i in prange(P):
        denom = 0.0

        for j in range(P):
            if i == j:
                continue

            diff = 0
            for k in range(N):
                if population[i, k] != population[j, k]:
                    diff += 1

            d = diff / N
            if d < sigma:
                denom += sharing_function_numba(d, sigma, alpha)

        # protect against division by zero
        if denom > 0.0:
            shared_fitness[i] = fitness[i] / denom
        else:
            shared_fitness[i] = fitness[i]
    return shared_fitness


@njit(parallel=True)
def all_islands_diversity_numba(populations):
    num_islands = len(populations)
    divs = np.empty(num_islands)
    for i in prange(num_islands):
        divs[i] = global_hamming_diversity_numba(populations[i])
    return divs
