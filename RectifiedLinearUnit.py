import numpy as np
import matplotlib.pyplot as plt

# Define the ReLU Activation Function
def relu(x):
    return np.maximum(0, x)

# Generate input values
x = np.linspace(-10, 10, 1000)
y = relu(x)

# Plot the ReLU Activation Function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = max(0, x)', color='orange')
plt.title('ReLU Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (f(x))')
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
