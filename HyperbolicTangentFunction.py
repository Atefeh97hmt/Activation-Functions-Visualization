import numpy as np
import matplotlib.pyplot as plt

# Define the Tanh Activation Function
def tanh(x):
    return np.tanh(x)

# Generate input values
x = np.linspace(-10, 10, 1000)
y = tanh(x)

# Plot the Tanh Activation Function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = tanh(x)', color='purple')
plt.title('Tanh Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (f(x))')
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
