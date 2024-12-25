import numpy as np
import matplotlib.pyplot as plt

# Define the Linear Activation Function
def linear_activation(x):
    return x

# Generate input values
x = np.linspace(-10, 10, 100)
y = linear_activation(x)

# Plot the Linear Activation Function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='f(x) = x', color='b')
plt.title('Linear Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Output (f(x))')
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
