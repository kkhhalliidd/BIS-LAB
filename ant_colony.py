import numpy as np
import random

dist_matrix = np.array([
    [0, 2, 2, 5, 7],
    [2, 0, 4, 8, 2],
    [2, 4, 0, 1, 3],
    [5, 8, 1, 0, 2],
    [7, 2, 3, 2, 0]
])

num_ants = 10
num_cities = len(dist_matrix)
alpha = 1
beta = 2
rho = 0.5
Q = 100
max_iter = 50

pheromone = np.ones((num_cities, num_cities))
best_cost = float('inf')
best_path = None

def path_length(path):
    return sum(dist_matrix[path[i % num_cities], path[(i + 1) % num_cities]] for i in range(num_cities))

for iteration in range(max_iter):
    all_paths = []
    all_costs = []

    for ant in range(num_ants):
        start = random.randint(0, num_cities - 1)
        path = [start]
        unvisited = list(range(num_cities))
        unvisited.remove(start)

        while unvisited:
            i = path[-1]
            probs = []
            for j in unvisited:
                tau = pheromone[i][j] ** alpha
                eta = (1 / dist_matrix[i][j]) ** beta
                probs.append(tau * eta)
            probs = np.array(probs) / np.sum(probs)
            next_city = random.choices(unvisited, weights=probs)[0]
            path.append(next_city)
            unvisited.remove(next_city)

        cost = path_length(path)
        all_paths.append(path)
        all_costs.append(cost)

        if cost < best_cost:
            best_cost = cost
            best_path = path

    pheromone *= (1 - rho)
    for k, path in enumerate(all_paths):
        for i in range(num_cities):
            a, b = path[i % num_cities], path[(i + 1) % num_cities]
            pheromone[a][b] += Q / all_costs[k]
            pheromone[b][a] = pheromone[a][b]

print("Best Path:", best_path)
print("Best Cost:", best_cost)
