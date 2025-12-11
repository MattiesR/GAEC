import Reporter
import numpy as np
from numba import njit, prange
import optuna

class Diagnostics:
	def __init__(self, enabled=False):
		self.enabled = enabled
		if enabled:
			self.data = {
				"diversity": [],
				"mutation_success": [],
				"crossover_success": []
			}

	def record(self, **kwargs):
		if not self.enabled:
			return
		for k, v in kwargs.items():
			self.data[k].append(v)




class r0877229:
	# -------------------
	# Hyperparameters
	# -------------------
	""" Population params """
	population_size = 200

	""" Variation params """
	crossover_rate = 0.85
	mutation_rate = 0.3
	mutation_patience = 50
	mutation_increase = 0.05	
	mut_high = 0.7
	mut_low = 0.3


	""" Initialization params """
	init_random_ratio = 0.0
	init_greedy_ratio = 1.0
	init_bfs_ratio = 0.0
	init_dfs_ratio = 0.0
	init_vectorized_random_ratio = 0.0
	""" Selection params """
	k_tournament = 2
	elitism_ratio = 0.001	# Default as 5%
	
	""" Variation params """
	
	# Ratios Mutations (weights of mutation schemes)
	swap_ratio = 0.55
	inversion_ratio = 0.35	
	scramble_ratio = 0.10 	# Occasional low probability

	""" Local search params """
	local_search_probability = 0.3
	K_lso = 12				# Number of neirest_neighbours
	max_improvement_lso = 20

	""" Island diversity"""
	island_diversity_init = 0.2
	island_diversity_rules = 0.2
	
	# -------------------
	# Objective function
	# -------------------
	best_objective = np.inf
	mean_objective = np.inf
	

	def __init__(self, filename=None):
		# Global-only hyperparameters (instance attributes) 
		""" Stopping criterea params"""
		self.max_iterations = 1000
		self.patience = 100


		""" Diversity promotion """
		self.num_islands = 4				 # 8
		self.migration_interval = 40	 # 100

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

		""" HERE REMOVE IT"""
		for i, isl in enumerate(islands):
			isl.mut_high = 0.5+(i*0.1)
		""" ------------- """
		iteration = 0
		no_improvement = 0

		while iteration < self.max_iterations:

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
			for idx, (_, obj) in enumerate(best_per_island):
				print(f"  Island {idx}: best objective = {obj}")

			# Stopping criteria
			if time_left < 0:
				break
			if no_improvement >= self.patience:
				break
			if best_objective < self.best_objective:
				no_improvement = 0
				self.mutation_rate = self.mut_low

			iteration += 1
			no_improvement += 1

			# Updating best objectives
			self.best_objective = best_objective
			self.mean_objective = mean_objective

			print(f"Iteration: {iteration}, best = {best_objective}, mean= {mean_objective}")


			# # Example: compute matrix of inter-island distances
			# num_islands = len(populations)
			# dist_matrix = np.zeros((num_islands, num_islands))
			# for i in range(num_islands):
			# 	for j in range(i+1, num_islands):
			# 		d = island_distance(populations[i], populations[j])
			# 		dist_matrix[i,j] = d

			# print("Inter-island distance matrix:")
			# print(dist_matrix)

		return 0



	# -------------------
	# GA Methods
	# -------------------
	"""Initialization algorithms"""
	def initialize_population(self, num_cities, pop_size, distance_matrix=None):
		"""
		Initialize the population using multiple strategies.
		Strategies and ratios are defined as class attributes:
			self.init_methods = [
				("random", self.init_random, self.init_random_ratio),
				("greedy", self.init_greedy, self.init_greedy_ratio),
				("bfs", self.init_graph_bfs, self.init_bfs_ratio),
				("dfs", self.init_graph_dfs, self.init_dfs_ratio),
			]
		"""

		# Build the list of (method, ratio) dynamically
		methods = [
			(self.init_random, self.init_random_ratio),
			(self.init_greedy, self.init_greedy_ratio),
			(self.init_graph_bfs, self.init_bfs_ratio),
			(self.init_graph_dfs, self.init_dfs_ratio),
			(self.init_vectorized_random, self.init_vectorized_random_ratio)
		]
		# Compute number of individuals per method
		counts = [int(pop_size * ratio) for _, ratio in methods]

		# Fix rounding to make total exactly pop_size
		remaining = pop_size - sum(counts)
		if remaining != 0:
			counts[0] += remaining  # Add the difference to the first method (random)

		""" Print statements"""
		print("------------------------------")
		print(f"Initialized population of {pop_size} individuals.")
		method_names = ["Random", "Greedy", "BFS", "DFS", "random_feasible"]
		for method, count in zip(method_names, counts):
			print(f"{method}: {count}")
		print("------------------------------")
		
		"""	-------------- """
		# Allocate population array
		population = np.zeros((pop_size, num_cities), dtype=np.int32)

		start_idx = 0
		for (method, _), count in zip(methods, counts):
			if count > 0:
				population[start_idx:start_idx+count] = method(
					distance_matrix if method != self.init_random else num_cities,	# Construction due to init_random taking other arguments
					count
				)
				start_idx += count

		# Shuffle rows to remove ordering bias
		np.random.shuffle(population)
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
		num_individuals = len(population)
		new_pop = np.zeros_like(population)

		# 1) Preserve top 'elitism' individuals
		elitism = int(self.population_size * self.elitism_ratio)
		if elitism > 0:
			elite_idx = np.argpartition(fitness, elitism)[:elitism]  # best fitness first
			new_pop[:elitism] = population[elite_idx]

		# 2) Fill rest of population
		for i in range(elitism, num_individuals):
			parent1, parent2 = self.select_parents(population, fitness)
			child = self.crossover(parent1, parent2)
			child = self.mutate(child)

			if np.random.rand() < self.local_search_probability:	# Apply LSO to children 
				N = distance_matrix.shape[0]
				candidate_list = np.zeros((N, self.K_lso), dtype=np.int32)
				for j in range(N):
					candidate_list[j] = np.argpartition(distance_matrix[j],self.K_lso)[:self.K_lso]
				child = two_opt_fast(child, distance_matrix, self.max_improvement_lso)
			new_pop[i] = child

		# 3) Elimination step
		combined_pop = np.vstack([population, new_pop])

		# Compute fitness for offspring (parent fitness known)
		offspring_fitness = self.evaluate_population(new_pop, distance_matrix)
		combined_fitness = np.concatenate((fitness, offspring_fitness))

		best_indices = np.argpartition(combined_fitness,num_individuals)[:num_individuals]
		# Eliminate the lambda worst => keep lambda best
		new_pop = combined_pop[best_indices]
		new_fitness = combined_fitness[best_indices]
		return new_pop, new_fitness

	""" Selection process """
	""" k-tournament selection (vectorized, faster for large populations) """

	def select_parents(self, population, fitness):
		"""
		Select two parents using k-tournament selection (vectorized with NumPy).
		Returns copies of parents.
		"""

		""" 
			Mixed strategy requires sample with probability from a method 
			Or by integrating directly in next_generation	
		"""
		def tournament():
			# Choose k random individuals
			idx = np.random.choice(len(population), self.k_tournament, replace=False)
			# Pick the one with lowest fitness (TSP: lower cost is better)
			best_idx = idx[np.argmin(fitness[idx])]
			return population[best_idx]

		parent1 = tournament()
		parent2 = tournament()
		return parent1.copy(), parent2.copy()



	""" Variation steps """
	def crossover(self, parent1, parent2):
		if np.random.rand() < self.crossover_rate:
			# return ordered_crossover(parent1, parent2)
			return epx_crossover(parent1, parent2)
			# return self.edge_recombination(parent1, parent2)
			# return erx_fast(parent1,parent2)
		return parent1.copy()
		
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

	def initialize(self, distance_matrix):
		num_cities = distance_matrix.shape[0]
		self.population = self.initialize_population(num_cities, self.population_size, distance_matrix)
		self.fitness = self.evaluate_population(self.population, distance_matrix)

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
		self.K_lso = max(1, int(self.K_lso * (1 + 0.2 * diversity_scale)))
		self.max_improvement_lso = max(1, int(self.max_improvement_lso * (1 + 0.2 * diversity_scale)))




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

@njit
def swap_mutation(individual):
    """Swap two random positions in the individual."""
    a, b = np.random.randint(0, len(individual), 2)
    individual[a], individual[b] = individual[b], individual[a]
    return individual

@njit
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

@njit
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

@njit
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
				js = range(i + 2, N)  # standard full loop
			else:
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
@njit(cache=True)
def init_greedy_numba(distance_matrix, pop_size, N, noise_scale):
    population = np.zeros((pop_size, N), dtype=np.int32)

    # We use prange, so calculations for each route must be independent.
    for k in prange(pop_size):
        
        # 1. Thread-safe Noise Calculation (local to this thread/route)
        if noise_scale > 0.0:
            # Generate a noise multiplier (e.g., in [0.99, 1.01])
            noise = np.random.uniform(1.0 - noise_scale, 1.0 + noise_scale, size=distance_matrix.shape)
            noisy_matrix = distance_matrix * noise
        else:
            noisy_matrix = distance_matrix
            
        # 2. Greedy construction logic
        current = 0 
        route = np.empty(N, dtype=np.int32)
        route[0] = current

        visited = np.zeros(N, dtype=np.bool_)
        visited[current] = True
        
        # Loop to fill the rest of the route
        for i in range(1, N):
            min_dist = np.inf
            next_city = -1
            
            # Find unvisited city with minimum distance from current
            for city in range(N):
                if not visited[city]:
                    # Use the noisy matrix for distance
                    dist = noisy_matrix[current, city]
                    if dist < min_dist:
                        min_dist = dist
                        next_city = city
            
            # The standard greedy algorithm assumes next_city is always found
            if next_city != -1:
                route[i] = next_city
                visited[next_city] = True
                current = next_city
            else:
                # Fallback only happens if all N-1 cities have been chosen.
                # Since the loop runs N-1 times, we just break here.
                break 
        population[k] = route
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