import Reporter
import numpy as np
import time
import multiprocessing as mp

# Modify the class name to match your student number.
class r0123456:
    def __init__(self, ouptut_file, mutation_scheme='inversion', base_mutation_rate=0.3):
        self.reporter = Reporter.Reporter(filename=ouptut_file)
        # mutation_scheme: 'swap', 'inversion', 'scramble', or 'random' (choose per application)
        self.mutation_scheme = mutation_scheme
        self.base_mutation_rate = base_mutation_rate

    # Ordered crossover (OX)
    def ordered_crossover(self, parent_a, parent_b, rng):
        """
        Performs ordered crossover (OX) between two parent permutations.
        Args:
            parent_a (np.ndarray): First parent permutation.
            parent_b (np.ndarray): Second parent permutation.
            rng (np.random.Generator): Random number generator.
        Returns:
            np.ndarray: Child permutation resulting from crossover.
        """
        n = parent_a.size
        child = -np.ones(n, dtype=int)
        i, j = sorted(rng.choice(n, size=2, replace=False))
        child[i:(j+1)] = parent_a[i:(j+1)]
        ptr = 0
        for val in parent_b:
            if val in child:
                continue
            while child[ptr] != -1:
                ptr += 1
            child[ptr] = val
        return child

    # Swap mutation (two positions swapped)
    def swap_mutation(self, individual, rng):
        n = individual.size
        i, j = rng.choice(n, size=2, replace=False)
        mutated = individual.copy()
        mutated[i], mutated[j] = mutated[j], mutated[i]
        return mutated

    # Inversion mutation (reverse subsequence between two cut points)
    def inversion_mutation(self, individual, rng):
        n = individual.size
        i, j = sorted(rng.choice(n, size=2, replace=False))
        mutated = individual.copy()
        mutated[i:(j+1)] = mutated[i:(j+1)][::-1]
        return mutated

    # Scramble mutation (shuffle elements within a subsequence)
    def scramble_mutation(self, individual, rng):
        n = individual.size
        i, j = sorted(rng.choice(n, size=2, replace=False))
        mutated = individual.copy()
        subseq = mutated[i:(j+1)].copy()
        rng.shuffle(subseq)
        mutated[i:(j+1)] = subseq
        return mutated

    # Apply the selected mutation scheme (or choose randomly if 'random')
    def apply_mutation(self, individual, rng):
        scheme = self.mutation_scheme
        if scheme == 'random':
            scheme = rng.choice(['swap', 'inversion', 'scramble'])
        if scheme == 'swap':
            return self.swap_mutation(individual, rng)
        if scheme == 'inversion':
            return self.inversion_mutation(individual, rng)
        if scheme == 'scramble':
            return self.scramble_mutation(individual, rng)
        # fallback to swap
        print("Warning: unknown mutation scheme '{}', defaulting to swap".format(scheme))
        return self.swap_mutation(individual, rng)

    # Compute tour length for permutation (permutation is nodes 1..N-1)
    def tour_length(self, perm, dist):
        # perm: numpy array of nodes excluding 0, values in 1..N-1
        # dist: full NxN distance matrix
        if perm.size == 0:
            # trivial: 0 -> 0
            return 0.0
        total = 0.0
        # 0 -> first
        d = dist[0, perm[0]]
        if np.isinf(d):
            return np.inf
        total += d
        # between nodes
        for a, b in zip(perm[:-1], perm[1:]):
            d = dist[a, b]
            if np.isinf(d):
                return np.inf
            total += d
        # last -> 0
        d = dist[perm[-1], 0]
        if np.isinf(d):
            return np.inf
        total += d
        return total

    # k-tournament selection (returns index)
    def tournament_select(self, objectives, k, rng):
        pop = objectives.size
        idxs = rng.integers(0, pop, size=k)
        best_idx = idxs[0]
        best_val = objectives[best_idx]
        for ii in idxs[1:]:
            if objectives[ii] < best_val:
                best_val = objectives[ii]
                best_idx = ii
        return best_idx

    # The evolutionary algorithm's main loop
    def optimize(self, filename, seed=42):
        file = open(filename)
        distanceMatrix = np.loadtxt(file, delimiter=",")
        file.close()

        rng = np.random.default_rng(seed=seed)

        N = distanceMatrix.shape[0]
        rep_size = N - 1

        

        POP_SIZE = 200
        K_TOURN = 5
        BASE_MUTATION_RATE = self.base_mutation_rate
        MUTATION_RATE = BASE_MUTATION_RATE
        MAX_MUTATION_RATE = 0.9
        MUTATION_INCREASE = 0.05
        ELITE_COUNT = 2
        MAX_GENERATIONS = 5000
        NO_IMPROVE_GENS = 300

        population = np.empty((POP_SIZE, rep_size), dtype=int)
        objectives = np.empty(POP_SIZE, dtype=float)
        nodes = np.arange(1, N, dtype=int)

        # Initial population
        for i in range(POP_SIZE):
            attempts = 0
            obj = np.inf
            while attempts < 50:
                perm = rng.permutation(nodes)
                obj = self.tour_length(perm, distanceMatrix)
                if not np.isinf(obj):
                    break
                attempts += 1
            if attempts >= 50 and np.isinf(obj):
                perm = rng.permutation(nodes)
                obj = self.tour_length(perm, distanceMatrix)
            population[i, :] = perm
            objectives[i] = obj

        bestObjective = np.min(objectives)
        bestSolutionPerm = population[np.argmin(objectives)].copy()
        meanObjective = np.mean(objectives[np.isfinite(objectives)]) if np.any(np.isfinite(objectives)) else np.inf

        gen = 0
        no_improve = 0
        # Main loop
        while True:
            # Report current stats
            # bestSolution must be cycle notation starting from 0 (we use array [0, perm...])
            bestSolution = np.concatenate(([0], bestSolutionPerm)).astype(int)
            timeLeft = self.reporter.report(meanObjective, bestObjective, bestSolution)
            if timeLeft < 0:
                break

            # Stopping criteria
            if gen >= MAX_GENERATIONS:
                break
            if no_improve >= NO_IMPROVE_GENS:
                break

            # Elitism: keep top ELITE_COUNT individuals
            sorted_idx = np.argsort(objectives)
            elites = population[sorted_idx[:ELITE_COUNT]].copy()
            elites_obj = objectives[sorted_idx[:ELITE_COUNT]].copy()

            # Produce offspring to fill remainder
            offspring = np.empty((POP_SIZE - ELITE_COUNT, rep_size), dtype=int)
            offspring_obj = np.empty(POP_SIZE - ELITE_COUNT, dtype=float)
            off_i = 0
            while off_i < POP_SIZE - ELITE_COUNT:
                # parent selection
                p1_idx = self.tournament_select(objectives, K_TOURN, rng)
                p2_idx = self.tournament_select(objectives, K_TOURN, rng)
                parent_a = population[p1_idx]
                parent_b = population[p2_idx]
                # crossover
                child = self.ordered_crossover(parent_a, parent_b, rng)
                # mutation
                if rng.random() < MUTATION_RATE:
                    child = self.apply_mutation(child, rng)

                    if np.array_equal(child, parent_a) or np.array_equal(child, parent_b):
                        child = self.apply_mutation(child, rng)

                # evaluate child
                obj = self.tour_length(child, distanceMatrix)
                offspring[off_i] = child
                offspring_obj[off_i] = obj
                off_i += 1

            # Form new population: elites + offspring
            population[:ELITE_COUNT] = elites
            objectives[:ELITE_COUNT] = elites_obj
            population[ELITE_COUNT:] = offspring
            objectives[ELITE_COUNT:] = offspring_obj

            # update statistics
            gen += 1
            current_best_idx = np.argmin(objectives)
            current_best = objectives[current_best_idx]
            current_best_perm = population[current_best_idx].copy()
            if current_best < bestObjective:
                bestObjective = current_best
                bestSolutionPerm = current_best_perm
                no_improve = 0
                MUTATION_RATE = BASE_MUTATION_RATE  # Reset mutation rate on improvement
            else:
                no_improve += 1
                # Increase mutation rate when no improvement
                MUTATION_RATE = min(MUTATION_RATE + MUTATION_INCREASE, MAX_MUTATION_RATE)

            mean_vals = objectives[np.isfinite(objectives)]
            meanObjective = np.mean(mean_vals) if mean_vals.size > 0 else np.inf

        # Final report once more before exit
        bestSolution = np.concatenate(([0], bestSolutionPerm)).astype(int)
        self.reporter.report(meanObjective, bestObjective, bestSolution)

        return 0


# Change here
tour_number = "50"
filename = f"/Users/julius/Library/CloudStorage/GoogleDrive-juliusjacobitz@gmail.com/My Drive/Studium/Master/07_Semester_Leuven/Genetic Algorithms/CodeGroupPhase/src/data/tour{tour_number}.csv"
folder = f"/Users/julius/Library/CloudStorage/GoogleDrive-juliusjacobitz@gmail.com/My Drive/Studium/Master/07_Semester_Leuven/Genetic Algorithms/CodeGroupPhase/src/data/output_julius/{tour_number}/"

def run_optimization(args):
    seed, tour_number, filename, folder = args
    solver = r0123456(ouptut_file=folder+f"tour_{tour_number}_seed{seed}_"+str(int(time.time())).split(".")[0])
    solver.optimize(filename, seed=seed)

if __name__ == '__main__':
    args_list = [(seed, tour_number, filename, folder) for seed in range(1, 101)]
    
    with mp.Pool(processes=mp.cpu_count()) as pool:
        pool.map(run_optimization, args_list)
