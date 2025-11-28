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
	population_size = 50

	""" Variation params """
	crossover_rate = 0.8
	mutation_rate = 0.2

	""" Stopping criterea params"""
	max_iterations = 1000
	patience = 50

	""" Initialization params """
	init_random_ratio = 0.5
	init_greedy_ratio = 0.3
	init_bfs_ratio = 0.1
	init_dfs_ratio = 0.1
	
	# -------------------
	# Objective function
	# -------------------
	best_objective = np.inf
	mean_objective = np.inf
	
	#--------------------
	# Diagnostic flags
	#--------------------
	DIAGNOSE = False


	def __init__(self, filename=None):
		if filename is None:
			filename = self.__class__.__name__
		self.reporter = Reporter.Reporter(filename)
	
	def enable_diagnostics(self):
		self.DIAGNOSE = True
		self.diag = Diagnostics(self.DIAGNOSE)

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

		# Initialize population
		population = self.initialize_population(len(distance_matrix), self.population_size, distance_matrix)

		iteration = 0
		no_improvement = 0
		
		while iteration < self.max_iterations:
			# Evaluate population
			fitness = self.evaluate_population(population, distance_matrix)

			# Reporting
			mean_objective = np.mean(fitness)
			best_idx = np.argmin(fitness)
			best_objective = fitness[best_idx]
			best_solution = population[best_idx] - 1  # Convert to 0-based for reporter

			time_left = self.reporter.report(mean_objective, best_objective, best_solution)

			# Genetic operations
			population = self.next_generation(population, fitness)

			# Stopping criteria
			if time_left < 0:
				break
			if no_improvement >= self.patience:
				break
			if best_objective < self.best_objective:
				no_improvement = 0

			iteration += 1
			no_improvement += 1

			# Updating best objectives
			self.best_objective = best_objective
			self.mean_objective = mean_objective

			# Diagnostic calculations
			if self.DIAGNOSE:
				diversity = compute_diversity(population)
				mutation_success = None
				crossover_success = None
				self.diag.record(
					diversity=diversity,
					mutation_success=mutation_success,
					crossover_success=crossover_success
				)

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
		]

		# Compute number of individuals per method
		counts = [int(pop_size * ratio) for _, ratio in methods]

		# Fix rounding to make total exactly pop_size
		remaining = pop_size - sum(counts)
		if remaining != 0:
			counts[0] += remaining  # Add the difference to the first method (random)

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
			pop[i] = np.random.permutation(num_cities) + 1
		return pop

	# Greedy 
	def init_greedy(self, distance_matrix, pop_size):
		num_cities = distance_matrix.shape[0]
		population = np.zeros((pop_size, num_cities), dtype=np.int32)

		for k in range(pop_size):
			current = np.random.randint(0, num_cities)
			visited = [current]
			unvisited = set(range(num_cities))
			unvisited.remove(current)

			while unvisited:
				next_city = min(unvisited, key=lambda j: distance_matrix[current, j])
				visited.append(next_city)
				unvisited.remove(next_city)
				current = next_city

			population[k] = np.array(visited, dtype=np.int32) + 1

		return population


	# --- Graph-aware Randomized DFS ---
	def init_graph_dfs(self, distance_matrix, pop_size):
		num_cities = distance_matrix.shape[0]
		population = np.zeros((pop_size, num_cities), dtype=np.int32)

		for k in range(pop_size):
			start = np.random.randint(0, num_cities)
			visited = [False] * num_cities
			path = []

			stack = [start]
			while stack:
				node = stack.pop()
				if not visited[node]:
					visited[node] = True
					path.append(node)

					# Neighbors: nodes with finite distance
					neighbors = [j for j in range(num_cities)
								if distance_matrix[node, j] != np.inf and not visited[j]]

					np.random.shuffle(neighbors)  # Randomize DFS traversal
					stack.extend(neighbors)

			population[k] = np.array(path, dtype=np.int32) + 1  # 1-based indexing

		return population


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

			population[k] = np.array(path, dtype=np.int32) + 1  # 1-based indexing

		return population


	""" Evalulation of the population """
	def evaluate_population(self, population, distance_matrix):
		return evaluate_population_numba(population, distance_matrix)

	""" next generation """
	def next_generation(self, population, fitness):
		new_pop = np.zeros_like(population)
		for i in range(len(population)):
			parent1, parent2 = self.select_parents(population, fitness)
			child = self.crossover(parent1, parent2)
			child = self.mutate(child)
			new_pop[i] = child
		return new_pop

	""" Selection process """
	def select_parents(self, population, fitness):
		idx1, idx2 = np.random.choice(len(population), 2, replace=False)
		parent1 = population[idx1] if fitness[idx1] < fitness[idx2] else population[idx2]

		idx3, idx4 = np.random.choice(len(population), 2, replace=False)
		parent2 = population[idx3] if fitness[idx3] < fitness[idx4] else population[idx4]

		return parent1, parent2


	""" Variation steps """
	def crossover(self, parent1, parent2):
		if np.random.rand() < self.crossover_rate:
			return ordered_crossover(parent1, parent2)
		return parent1.copy()

	def mutate(self, individual):
		if np.random.rand() < self.mutation_rate:
			return swap_mutation(individual)
		return individual


# -------------------
# Diagnostic functions
# -------------------
def compute_diversity(population):
	return np.mean([
		np.sum(population[i] != population[j])
		for i in range(len(population))
		for j in range(i+1, len(population))
	])


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
			from_city = tour[j] - 1
			to_city = tour[(j + 1) % len(tour)] - 1
			total += distance_matrix[from_city, to_city]
		fitness[i] = total
	return fitness

@njit
def swap_mutation(individual):
	a, b = np.random.randint(0, len(individual), 2)
	individual[a], individual[b] = individual[b], individual[a]
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
