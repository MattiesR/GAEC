import r0877229



if __name__ == "__main__":
	filename = "diagnostic"
	solver = r0877229.r0877229(filename)
	solver.enable_diagnostics()
	solver.optimize("src/data/tour50.csv")
	print(solver.diag.data["diversity"])
