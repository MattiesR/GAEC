import Reporter
import numpy as np
from numba import njit, prange
import optuna

class r0877229:

    # -------------------
    # Hyperparameters
    # -------------------
    population_size = 50
    crossover_rate = 0.8
    mutation_rate = 0.2
    max_iterations = 100

	# -------------------
    # Objective function
    # -------------------
    best_objective = None
    mean_objective = None
    

    def __init__(self, filename=None):
        if filename is None:
            filename = self.__class__.__name__
        self.reporter = Reporter.Reporter(filename)

    # -------------------
    # Main optimization loop
    # -------------------
    def optimize(self, filename):
        distance_matrix = np.loadtxt(filename, delimiter=",")

        # Initialize population
        population = self.initialize_population(len(distance_matrix), self.population_size)

        iteration = 0
        while iteration < self.max_iterations:
            # Evaluate population
            fitness = self.evaluate_population(population, distance_matrix)

            # Reporting
            mean_objective = np.mean(fitness)
            best_idx = np.argmin(fitness)
            best_objective = fitness[best_idx]
            best_solution = population[best_idx] - 1  # Convert to 0-based for reporter

            time_left = self.reporter.report(mean_objective, best_objective, best_solution)
            
			# Updating best objectives
            self.best_objective = best_objective
            self.mean_objective = mean_objective
            if time_left < 0:
                break

            # Genetic operations
            population = self.next_generation(population, fitness)

            iteration += 1

        return 0

    # -------------------
    # GA Methods
    # -------------------
    def initialize_population(self, num_cities, pop_size):
        pop = np.zeros((pop_size, num_cities), dtype=np.int32)
        for i in range(pop_size):
            pop[i] = np.random.permutation(num_cities) + 1  # 1-based
        return pop

    def evaluate_population(self, population, distance_matrix):
        # Use numba-accelerated function
        return evaluate_population_numba(population, distance_matrix)

    def next_generation(self, population, fitness):
        # Selection, crossover, mutation
        new_pop = np.zeros_like(population)
        for i in range(len(population)):
            parent1, parent2 = self.select_parents(population, fitness)
            child = self.crossover(parent1, parent2)
            child = self.mutate(child)
            new_pop[i] = child
        return new_pop

    def select_parents(self, population, fitness):
        # Tournament selection
        idx1, idx2 = np.random.choice(len(population), 2, replace=False)
        parent1 = population[idx1] if fitness[idx1] < fitness[idx2] else population[idx2]
        idx3, idx4 = np.random.choice(len(population), 2, replace=False)
        parent2 = population[idx3] if fitness[idx3] < fitness[idx4] else population[idx4]
        return parent1, parent2

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            return ordered_crossover(parent1, parent2)
        return parent1.copy()

    def mutate(self, individual):
        if np.random.rand() < self.mutation_rate:
            return swap_mutation(individual)
        return individual

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


def objective(trial):
    solver = r0877229()
    solver.population_size = trial.suggest_int("pop_size", 100, 200)
    solver.mutation_rate = trial.suggest_float("mut_rate", 0.01, 0.5)
    solver.crossover_rate = trial.suggest_float("cross_rate", 0.5, 1.0)
    solver.max_iterations = 50
    solver.optimize("./src/data/tour50.csv")
    
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
    