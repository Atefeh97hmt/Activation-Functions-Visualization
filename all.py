import numpy as np
import matplotlib.pyplot as plt

# Define activation functions
def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, x * alpha)

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return exp_x / np.sum(exp_x)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu(x):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))

def softplus(x):
    return np.log(1 + np.exp(x))

def gelu(x):
    return x * (1 / (1 + np.exp(-1.702 * x)))

def swish(x):
    return x * sigmoid(x)

def mish(x):
    return x * np.tanh(softplus(x))

# Generate input data
x = np.linspace(-5, 5, 1000)
x_softmax = np.linspace(-2, 2, 1000)  # Smaller range for softmax

# Create subplots
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()

# Plot each activation function
functions = [
    (linear, "Linear", x),
    (sigmoid, "Sigmoid", x),
    (tanh, "Tanh", x),
    (relu, "ReLU", x),
    (leaky_relu, "Leaky ReLU", x),
    (softmax, "Softmax", x_softmax),
    (elu, "ELU", x),
    (selu, "SELU", x),
    (softplus, "Softplus", x),
    (gelu, "GELU", x),
    (swish, "Swish", x),
    (mish, "Mish", x)
]

for i, (func, name, x_data) in enumerate(functions):
    y = func(x_data)
    axes[i].plot(x_data, y, label=name, color='b')
    axes[i].set_title(name)
    axes[i].set_xlabel('x')
    axes[i].set_ylabel(f'{name}(x)')
    axes[i].grid(True)
    axes[i].legend()
    axes[i].set_ylim(min(y) - 0.5, max(y) + 0.5)

# Adjust layout to prevent overlap
plt.tight_layout()

# Show plot
plt.show()