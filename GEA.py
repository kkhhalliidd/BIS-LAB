import cv2
import numpy as np
import random

def fitness_function(img, threshold):
    pixels = img.flatten()
    w0 = np.sum(pixels <= threshold) / len(pixels)
    w1 = np.sum(pixels > threshold) / len(pixels)
    if w0 == 0 or w1 == 0:
        return 0
    m0 = np.mean(pixels[pixels <= threshold])
    m1 = np.mean(pixels[pixels > threshold])
    return w0 * w1 * (m0 - m1) ** 2

img = cv2.imread("your_image.jpg", cv2.IMREAD_GRAYSCALE)

POP_SIZE = 20
GEN_MAX = 50
Pc = 0.8
Pm = 0.1

population = [random.randint(0, 255) for _ in range(POP_SIZE)]

for gen in range(GEN_MAX):
    fitness_values = [fitness_function(img, t) for t in population]
    total_fit = sum(fitness_values)
    probs = [f / total_fit if total_fit > 0 else 0 for f in fitness_values]
    selected = random.choices(population, weights=probs, k=POP_SIZE)

    new_population = []
    for i in range(0, POP_SIZE, 2):
        p1, p2 = selected[i], selected[i + 1]
        if random.random() < Pc:
            r = random.randint(0, 7)
            mask = (1 << r) - 1
            c1 = (p1 & ~mask) | (p2 & mask)
            c2 = (p2 & ~mask) | (p1 & mask)
        else:
            c1, c2 = p1, p2
        if random.random() < Pm:
            c1 = random.randint(0, 255)
        if random.random() < Pm:
            c2 = random.randint(0, 255)
        new_population.extend([c1, c2])

    population = new_population
    best_thresh = population[np.argmax([fitness_function(img, t) for t in population])]
    print(f"Generation {gen+1}: Best Threshold = {best_thresh}")

final_threshold = best_thresh
_, final_binary = cv2.threshold(img, final_threshold, 255, cv2.THRESH_BINARY)

cv2.imshow("Original Image", img)
cv2.imshow("GA Black & White Image", final_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nOptimal Threshold Found:", final_threshold)
