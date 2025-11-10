import random

values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50


POP_SIZE = 10
GEN_MAX = 50
Pc = 0.8
Pm = 0.1


def fitness(chromosome):
    total_value = 0
    total_weight = 0
    for i in range(len(chromosome)):
        if chromosome[i] == 1:
            total_value += values[i]
            total_weight += weights[i]
    if total_weight > capacity:
        return 0  # invalid solution
    return total_value

def selection(pop):
    total_fit = sum(fitness(ch) for ch in pop)
    probs = [fitness(ch) / total_fit for ch in pop]
    return random.choices(pop, weights=probs, k=2)

def crossover(p1, p2):
    if random.random() < Pc:
        r = random.randint(1, len(p1)-1)
        c1 = p1[:r] + p2[r:]
        c2 = p2[:r] + p1[r:]
        return c1, c2
    return p1, p2

def mutate(chrom):
    for i in range(len(chrom)):
        if random.random() < Pm:
            chrom[i] = 1 - chrom[i]
    return chrom


population = [[random.randint(0, 1) for _ in range(len(values))] for _ in range(POP_SIZE)]

 
best_solution = None
best_fitness = 0

for gen in range(GEN_MAX):
    new_pop = []
    for _ in range(POP_SIZE // 2):
        p1, p2 = selection(population)
        c1, c2 = crossover(p1, p2)
        c1 = mutate(c1)
        c2 = mutate(c2)
        new_pop.extend([c1, c2])
    population = new_pop

    for ch in population:
        fit = fitness(ch)
        if fit > best_fitness:
            best_fitness = fit
            best_solution = ch

print("Best Solution:", best_solution)
print("Total Value:", best_fitness)
print("Total Weight:", sum(weights[i] for i in range(len(weights)) if best_solution[i] == 1))
