import numpy as np
import matplotlib.pyplot as plt

# Define the Softmax Activation Function
def softmax(x):
    exp_x = np.exp(x - np.max(x))  # For numerical stability
    return exp_x / np.sum(exp_x)

# Generate input values (example logits)
x = np.linspace(-5, 5, 100)
y = softmax(x)

# Plot the Softmax Activation Function
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='Softmax Function', color='gray')
plt.title('Softmax Activation Function')
plt.xlabel('Input (x)')
plt.ylabel('Probability (f(x))')
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.show()
