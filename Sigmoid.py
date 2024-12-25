import numpy as np
import matplotlib.pyplot as plt

# Define the Sigmoid Activation Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate input values
x = np.linspace(-10, 10, 1000)
y = sigmoid(x)

# Plot the Sigmoid Activation Function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = 1 / (1 + e^(-x))', color='g')
plt.title('Sigmoid Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (f(x))')
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()