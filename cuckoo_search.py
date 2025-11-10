import random
import numpy as np

values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50
n = len(values)

num_nests = 30
pa = 0.25
max_iter = 50

def fitness(sol):
    total_w = sum(weights[i] for i in range(n) if sol[i] == 1)
    total_v = sum(values[i]  for i in range(n) if sol[i] == 1)
    if total_w > capacity:
        return 0
    return total_v

def random_solution():
    return [random.randint(0,1) for _ in range(n)]

def levy_flight(sol):
    new = sol.copy()
    i = random.randrange(n)
    new[i] = 1 - new[i]
    return new

nests = [random_solution() for _ in range(num_nests)]
fitnesses = [fitness(sol) for sol in nests]
best_nest = nests[np.argmax(fitnesses)]
best_fit  = max(fitnesses)

for it in range(max_iter):
    for i in range(num_nests):
        new_sol = levy_flight(nests[i])
        new_fit = fitness(new_sol)
        if new_fit > fitnesses[i]:
            nests[i] = new_sol
            fitnesses[i] = new_fit
    for i in range(int(pa * num_nests)):
        idx = fitnesses.index(min(fitnesses))
        nests[idx] = random_solution()
        fitnesses[idx] = fitness(nests[idx])
    current_best = max(fitnesses)
    if current_best > best_fit:
        best_fit  = current_best
        best_nest = nests[fitnesses.index(best_fit)]
    print(f"Iteration {it+1}: Best Value = {best_fit}")

print("Best solution:", best_nest)
print("Best value:", best_fit)
print("Total weight:", sum(weights[i] for i in range(n) if best_nest[i]==1))
