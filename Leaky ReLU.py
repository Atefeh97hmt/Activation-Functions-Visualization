import numpy as np
import matplotlib.pyplot as plt

# Define the Leaky ReLU Activation Function
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# Generate input values
x = np.linspace(-10, 10, 1000)
y = leaky_relu(x, alpha=0.01)

# Plot the Leaky ReLU Activation Function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = Leaky ReLU (α=0.01)', color='red')
plt.title('Leaky ReLU Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (f(x))')
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
