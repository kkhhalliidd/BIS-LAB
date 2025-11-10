import cv2
import numpy as np
import random

def fitness_function(img, T):
    pixels = img.flatten()
    w0 = np.sum(pixels <= T) / len(pixels)
    w1 = np.sum(pixels > T) / len(pixels)
    if w0 == 0 or w1 == 0:
        return 0
    m0 = np.mean(pixels[pixels <= T])
    m1 = np.mean(pixels[pixels > T])
    return w0 * w1 * (m0 - m1) ** 2

img = cv2.imread("your_image.jpg", cv2.IMREAD_GRAYSCALE)

num_wolves = 20
max_iter = 50
lower_bound = 0
upper_bound = 255

positions = np.random.randint(lower_bound, upper_bound + 1, num_wolves)
fitness_values = np.array([fitness_function(img, t) for t in positions])

alpha = positions[np.argmax(fitness_values)]
beta = positions[np.argsort(fitness_values)[-2]]
delta = positions[np.argsort(fitness_values)[-3]]

for t in range(max_iter):
    a = 2 * (1 - t / max_iter)
    for i in range(num_wolves):
        r1, r2 = random.random(), random.random()
        A1 = 2 * a * r1 - a
        C1 = 2 * r2
        D_alpha = abs(C1 * alpha - positions[i])
        X1 = alpha - A1 * D_alpha

        r1, r2 = random.random(), random.random()
        A2 = 2 * a * r1 - a
        C2 = 2 * r2
        D_beta = abs(C2 * beta - positions[i])
        X2 = beta - A2 * D_beta

        r1, r2 = random.random(), random.random()
        A3 = 2 * a * r1 - a
        C3 = 2 * r2
        D_delta = abs(C3 * delta - positions[i])
        X3 = delta - A3 * D_delta

        new_pos = int((X1 + X2 + X3) / 3)
        new_pos = max(lower_bound, min(upper_bound, new_pos))
        new_fit = fitness_function(img, new_pos)
        if new_fit > fitness_values[i]:
            positions[i] = new_pos
            fitness_values[i] = new_fit

    sorted_indices = np.argsort(fitness_values)
    alpha = positions[sorted_indices[-1]]
    beta = positions[sorted_indices[-2]]
    delta = positions[sorted_indices[-3]]
    print(f"Iteration {t+1}: Best Threshold = {alpha}")

best_threshold = alpha
_, segmented = cv2.threshold(img, best_threshold, 255, cv2.THRESH_BINARY)

cv2.imshow("Original Image", img)
cv2.imshow("GWO Segmented Image", segmented)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Optimal Threshold Found:", best_threshold)
