import cv2
import numpy as np
import random

# --- Objective function: between-class variance ---
def fitness_function(img, threshold):
    # Binarize image using threshold
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    # Compute means and probabilities
    pixels = img.flatten()
    w0 = np.sum(pixels <= threshold) / len(pixels)
    w1 = np.sum(pixels > threshold) / len(pixels)
    if w0 == 0 or w1 == 0:
        return 0
    m0 = np.mean(pixels[pixels <= threshold])
    m1 = np.mean(pixels[pixels > threshold])
    # Between-class variance
    return w0 * w1 * (m0 - m1) ** 2

# --- Load grayscale image ---
img = cv2.imread("your_image.jpg", cv2.IMREAD_GRAYSCALE)

# --- PSO Parameters ---
num_particles = 20
max_iter = 50
w = 0.7      
c1 = 1.5      
c2 = 1.5      


positions = np.random.randint(0, 256, num_particles)
velocities = np.zeros(num_particles)
pbest = positions.copy()
pbest_values = np.array([fitness_function(img, t) for t in positions])

gbest = pbest[np.argmax(pbest_values)]
gbest_value = np.max(pbest_values)


for it in range(max_iter):
    for i in range(num_particles):

        r1, r2 = random.random(), random.random()
        velocities[i] = (w * velocities[i] +
                         c1 * r1 * (pbest[i] - positions[i]) +
                         c2 * r2 * (gbest - positions[i]))
        positions[i] = int(np.clip(positions[i] + velocities[i], 0, 255))
        

        fit = fitness_function(img, positions[i])
        if fit > pbest_values[i]:
            pbest_values[i] = fit
            pbest[i] = positions[i]
            

    gbest = pbest[np.argmax(pbest_values)]
    gbest_value = np.max(pbest_values)

    print(f"Iteration {it+1}: Best Threshold = {gbest}")


_, final_binary = cv2.threshold(img, gbest, 255, cv2.THRESH_BINARY)

cv2.imshow("Original Image", img)
cv2.imshow("Optimized Black & White Image", final_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\nOptimal Threshold Found:", gbest)
