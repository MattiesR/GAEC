import numpy as np
import r0877229

import cProfile
import pstats
import plots
import pandas as pd

def check_tour_validity(filename, N=None):
    df = pd.read_csv(filename, comment="#", header=None)

    # Tours start at column 5 (0-based indexing)
    tours = df.iloc[:, 4:].to_numpy()  # exclude metadata
    pop_size, num_cols = tours.shape

    if N is None:
        N = num_cols  # number of cities

    for k in range(pop_size):
        tour = tours[k, :N].astype(int)  # ignore any extra trailing NaN

        # Check for -1 or NaN
        if np.any(tour < 0) or np.any(np.isnan(tour)):
            print(f"Invalid value in tour {k}: {tour}")
            raise ValueError

        # Check all cities are present exactly once
        if set(tour) != set(range(N)):
            missing = set(range(N)) - set(tour)
            duplicates = [x for x in tour if list(tour).count(x) > 1]
            print(f"Tour {k} is invalid! Missing: {missing}, Duplicates: {set(duplicates)}")
            raise ValueError
        # Warn if tour does not start with 0
        if tour[0] != 0:
            print(f"Warning: tour {k} does not start with 0: starts with {tour[0]}")
    print("Sanity check complete.")





# Warm up all Numba functions first
warmup_size = 15
print(f"WARMING UP WITH TOUR {warmup_size}")
solver = r0877229.r0877229(filename = f"geac_tour_{warmup_size}")
# solver.K_lso = 3
solver.optimize(f"src/data/tour{warmup_size}.csv")
print("WE WARMED UP")
# Now start profiling



size = 500
filename = f"src/data/tour{size}.csv"

solver = r0877229.r0877229(filename = f"geac_tour_{size}")


cProfile.run(f"solver.optimize('{filename}')", "profiler_output.prof")



stats = pstats.Stats("profiler_output.prof")
stats.strip_dirs()              # remove long paths
stats.sort_stats("cumtime")     # or "tottime"
stats.print_stats("r0877229.py")  # only show functions from your file
check_tour_validity(f"geac_tour_{size}.csv", size)

# plots.plot_convergence(f"geac_tour_{size}")