import r0877229
import pandas as pd
import matplotlib.pyplot as plt
import plots


if __name__ == "__main__":
	size = 750
	filename = f"geac_tour_{size}"
	solver = r0877229.r0877229(filename)
	solver.optimize(f"src/data/tour{size}.csv")
	plots.plot_convergence(filename)