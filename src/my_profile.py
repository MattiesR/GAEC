import numpy as np
import r0877229

import cProfile
import pstats
size = 750
filename = f"src/data/tour{size}.csv"





# Warm up all Numba functions first
warmup_size = 15
print(f"WARMING UP WITH TOUR {warmup_size}")
solver = r0877229.r0877229(filename = f"geac_tour_{warmup_size}")
# solver.K_lso = 3
solver.optimize(f"src/data/tour{warmup_size}.csv")
print("WE WARMED UP")
# Now start profiling


solver = r0877229.r0877229(filename)
cProfile.run(f"solver.optimize('{filename}')", "profiler_output.prof")



stats = pstats.Stats("profiler_output.prof")
stats.strip_dirs()              # remove long paths
stats.sort_stats("cumtime")     # or "tottime"
stats.print_stats("r0877229.py")  # only show functions from your file
