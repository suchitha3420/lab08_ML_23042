

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier

# Activation functions
def step_function(x):
    return 1 if x >= 0 else 0

def bipolar_step_function(x):
    return 1 if x >= 0 else -1

def sigmoid_function(x):
    return 1 / (1 + np.exp(-np.array(x)))

def tanh_function(x):
    return np.tanh(x)

def relu_function(x):
    return max(0, x)

def leaky_relu_function(x, alpha=0.01):
    return x if x > 0 else alpha * x

# Summation function
def summation(inputs, weights):
    return np.dot(inputs, weights)

# Error function
def squared_error(y_true, y_pred):
    return 0.5 * (y_true - y_pred) ** 2

# Perceptron training function
def perceptron_train(training_data, learning_rate=0.1, epochs=100):
    num_inputs = len(training_data[0][0])  # Number of input features
    weights = np.random.rand(num_inputs)  # Random weight initialization
    bias = np.random.rand()

    errors = []  # Track error reduction
    for epoch in range(epochs):
        total_error = 0
        for x_input, y_true in training_data:
            y_pred = step_function(summation(x_input, weights) + bias)
            error = y_true - y_pred
            weights += learning_rate * error * np.array(x_input)  # Update weights
            bias += learning_rate * error  # Update bias
            total_error += abs(error)

        errors.append(total_error)  # Store epoch error
        if total_error == 0:  # Stop early if no error
            break

    return weights, bias, errors

# AND Gate training
and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1)
]

weights, bias, errors = perceptron_train(and_data)
print(f"AND Gate Perceptron: Weights: {weights}, Bias: {bias}")

# Plot error reduction
plt.plot(errors)
plt.xlabel("Epochs")
plt.ylabel("Total Error")
plt.title("Error Reduction in AND Perceptron Training")
plt.show()

# XOR Gate using Perceptron (Incorrect)
xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

weights, bias, errors = perceptron_train(xor_data)
print(f"XOR Perceptron (Incorrect): Weights: {weights}, Bias: {bias}")
plt.plot(errors)
plt.xlabel("Epochs")
plt.ylabel("Total Error")
plt.title("XOR Training with Single-Layer Perceptron (Fails to Converge)")
plt.show()

# XOR Using MLPClassifier (Correct)
mlp = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', max_iter=5000, solver='adam')
X = np.array([x[0] for x in xor_data])
y = np.array([x[1] for x in xor_data])
mlp.fit(X, y)
print(f"XOR Prediction using MLP: {mlp.predict(X)}")

# Learning Rate vs. Epochs Experiment
learning_rates = [0.01, 0.1, 1]
plt.figure(figsize=(8, 6))
for lr in learning_rates:
    _, _, errors = perceptron_train(and_data, learning_rate=lr)
    plt.plot(errors, label=f"LR={lr}")

plt.xlabel("Epochs")
plt.ylabel("Total Error")
plt.title("Effect of Learning Rate on Convergence")
plt.legend()
plt.show()

# XOR Using Pseudo-Inverse
X = np.array([x[0] + [1] for x in xor_data])  # Add bias as last column
y = np.array([x[1] for x in xor_data])

W_pseudo = np.linalg.pinv(X).dot(y)
print(f"Pseudo-Inverse XOR Weights: {W_pseudo}")

# Prediction using Pseudo-Inverse
predictions = np.round(X @ W_pseudo)
print(f"XOR Predictions using Pseudo-Inverse: {predictions}")

