import numpy as np
import pandas as pd
import argparse

def generate_tsp(n, connectivity=0.8, max_distance=100):
    """
    Generates a TSP distance matrix with N cities.
    Some cities may be unconnected (distance = inf).

    Parameters:
    - n: number of cities
    - connectivity: probability that a city pair is connected
    - max_distance: maximum distance between connected cities

    Returns:
    - distance_matrix: n x n numpy array
    """
    distance_matrix = np.full((n, n), np.inf)
    
    for i in range(n):
        for j in range(i+1, n):
            if np.random.rand() < connectivity:
                distance = np.random.randint(1, max_distance+1)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # symmetric
    
    np.fill_diagonal(distance_matrix, 0)  # distance to self is 0
    return distance_matrix

def save_to_csv(matrix, filename):
    df = pd.DataFrame(matrix)
    df.to_csv(filename, index=False, header=False)
    print(f"TSP problem saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate TSP problem with N cities")
    parser.add_argument("N", type=int, help="Number of cities")
    args = parser.parse_args()
    N = args.N
    if N == 50 or N == 250 or N==500 or N== 750 or N ==1000:
        raise ValueError
    tsp_matrix = generate_tsp(N)
    output= f"tour{args.N}.csv"
    save_to_csv(tsp_matrix, output)
