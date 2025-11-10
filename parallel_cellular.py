import numpy as np
import random
import itertools

dist_matrix = np.array([
    [0, 2, 9, 10],
    [1, 0, 6, 4],
    [15, 7, 0, 8],
    [6, 3, 12, 0]
])

grid_size = (4, 4)
num_cities = len(dist_matrix)
pop_size = grid_size[0] * grid_size[1]
Pc = 0.8
Pm = 0.2
max_iter = 100

def path_length(path):
    return sum(dist_matrix[path[i], path[(i + 1) % num_cities]] for i in range(num_cities))

def random_path():
    p = list(range(num_cities))
    random.shuffle(p)
    return p

def crossover(p1, p2):
    if random.random() > Pc:
        return p1.copy(), p2.copy()
    a, b = sorted(random.sample(range(num_cities), 2))
    c1, c2 = [-1]*num_cities, [-1]*num_cities
    c1[a:b], c2[a:b] = p1[a:b], p2[a:b]
    fill_c1 = [x for x in p2 if x not in c1]
    fill_c2 = [x for x in p1 if x not in c2]
    idx1, idx2 = 0, 0
    for i in range(num_cities):
        if c1[i] == -1:
            c1[i] = fill_c1[idx1]
            idx1 += 1
        if c2[i] == -1:
            c2[i] = fill_c2[idx2]
            idx2 += 1
    return c1, c2

def mutate(path):
    if random.random() < Pm:
        i, j = random.sample(range(num_cities), 2)
        path[i], path[j] = path[j], path[i]
    return path

def get_neighbors(idx, grid_shape):
    x, y = divmod(idx, grid_shape[1])
    neighbors = []
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = (x+dx)%grid_shape[0], (y+dy)%grid_shape[1]
        neighbors.append(nx*grid_shape[1]+ny)
    return neighbors

population = [random_path() for _ in range(pop_size)]
fitness = [1 / path_length(p) for p in population]

for iteration in range(max_iter):
    new_population = []
    for i in range(pop_size):
        neighbors = get_neighbors(i, grid_size)
        local = [population[j] for j in neighbors]
        local_fitness = [fitness[j] for j in neighbors]
        p1 = local[np.argmax(local_fitness)]
        p2 = random.choice(local)
        c1, c2 = crossover(p1, p2)
        c1 = mutate(c1)
        c2 = mutate(c2)
        best_child = min([c1, c2], key=path_length)
        new_population.append(best_child)
    population = new_population
    fitness = [1 / path_length(p) for p in population]
    best_path = population[np.argmax(fitness)]
    best_cost = path_length(best_path)
    print(f"Iteration {iteration+1}: Best Cost = {best_cost}")

best_path = population[np.argmax(fitness)]
best_cost = path_length(best_path)
print("Best Path:", best_path)
print("Best Cost:", best_cost)
