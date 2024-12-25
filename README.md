# Activation Functions Visualization

This repository contains Python code that generates visualizations for various activation functions commonly used in neural networks. The activation functions included are:

1. **Linear Activation Function**
2. **Sigmoid Activation Function**
3. **Hyperbolic Tangent (Tanh) Activation Function**
4. **Rectified Linear Unit (ReLU) Activation Function**
5. **Leaky ReLU Activation Function**
6. **Softmax Activation Function**

These visualizations are useful for understanding how different activation functions behave with respect to input values. They help in visualizing their properties, such as non-linearity, saturation regions, and how they are used in neural network architectures.

---

## Functions Visualized:

### 1. **Linear Activation Function**
- Maps the input directly to the output with no transformation.  
  Formula: \( f(x) = x \)

### 2. **Sigmoid Activation Function**
- Outputs values between 0 and 1, commonly used in binary classification tasks.  
  Formula: \( f(x) = \frac{1}{1 + e^{-x}} \)

### 3. **Hyperbolic Tangent (Tanh) Activation Function**
- Outputs values between -1 and 1, making it zero-centered.  
  Formula: \( f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} \)

### 4. **Rectified Linear Unit (ReLU) Activation Function**
- Outputs the input if it's positive, otherwise, it outputs 0.  
  Formula: \( f(x) = \max(0, x) \)

### 5. **Leaky ReLU Activation Function**
- Similar to ReLU but allows a small, non-zero gradient for negative inputs.  
  Formula: \( f(x) = \alpha x \) if \( x \leq 0 \), else \( f(x) = x \)

### 6. **Softmax Activation Function**
- Converts a vector of raw scores (logits) into probabilities that sum to 1. Typically used in multi-class classification problems.  
  Formula: \( f(x_i) = \frac{e^{x_i}}{\sum_{j=1}^{n} e^{x_j}} \)

---

## Prerequisites:

To run the code, you need to have the following Python packages installed:
- `numpy`
- `matplotlib`

You can install them using pip:

```bash
pip install numpy matplotlib
