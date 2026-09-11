

import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

threshold = 0.5

sample_z_values = [-4, -1.5, -0.2, 0, 0.2, 1.8, 4]

for z in sample_z_values:
    prob = sigmoid(z)
    if prob >= threshold:
        prediction = 1
    else:
        prediction = 0
    print(f"z = {z:>5} -> sigmoid(z) = {prob:.4f} -> prediction = {prediction}")