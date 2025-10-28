import numpy as np
import torch
from matplotlib import pyplot as plt

# --- 1. Define the original distribution p(z) ---
# We define a uniform distribution over the interval [0, 1].
n_points = 1000
z_range = [0, 1]

# Create a grid of points to evaluate the PDF. We use a wider range for plotting purposes.
z_vals_grid = torch.linspace(-2, 12, n_points)

# Calculate the probability density p(z) on the grid.
# Initialize with zeros, as the density is 0 outside the support.
p_z = torch.zeros_like(z_vals_grid)
mask_z = (z_vals_grid >= z_range[0]) & (z_vals_grid <= z_range[1])
p_z[mask_z] = 1.0 / (z_range[1] - z_range[0])

# --- 2. Define the transformation y = g(z) ---
# We define a simple linear transformation: a shift by a constant 'c'.
a, b = 3.0, 7.0 # Let's use more distinct values for clarity
y_vals_grid = (b-a)*z_vals_grid+a

# --- 3. EXPLICITLY Calculate the Jacobian using Autograd ---
# The Change of Variables formula is: p_y(y) = p_z(g_inverse(y)) * |jacobian|
# To calculate the jacobian d(g_inverse(y))/dy, we need a tensor to differentiate with respect to.
# We use a clone of our y_vals grid and tell PyTorch to track its gradients for the calculation.
y_for_grad = y_vals_grid.clone().requires_grad_()

# Define the inverse transformation g_inverse(y) = y - c
z_from_y = (y_for_grad-a)/(b-a)

# THIS IS THE LINE THAT CALCULATES THE DERIVATIVE
# We use torch.autograd.grad to compute the gradient of `z_from_y` (our g_inverse)
# with respect to `y_for_grad` (our y).
jacobian = torch.autograd.grad(z_from_y, y_for_grad, torch.ones_like(y_for_grad))[0]

# --- 4. Calculate the new density p(y) using the full formula ---
# The formula is p_y(y) = p_z(g_inverse(y)) * |Jacobian|.
# Since p_z is 1 on its support, this simplifies to p_y(y) = 1 * |Jacobian|.
# The Jacobian for this transformation is a constant: 1 / (b - a).

# Initialize p_y with zeros.
p_y = torch.zeros_like(y_vals_grid)
# Find the new support for y, which is the interval [a, b].
mask_y = (y_vals_grid >= a) & (y_vals_grid <= b)
# Set the density within the new support. The height is the absolute value of the Jacobian.
p_y[mask_y] = torch.abs(jacobian[mask_y]) * p_z[mask_z]

# --- 5. Plot the results for comparison ---
plt.figure(figsize=(10, 6))
plt.plot(z_vals_grid.numpy(), p_z.numpy(), label='Original density p(z) ~ U(0,1)', color='blue', linewidth=2)
plt.plot(y_vals_grid.numpy(), p_y.numpy(), label=f'Transformed density p(y) ~ U({a:.1f},{b:.1f})', color='red', linestyle='--', linewidth=2)

plt.title("Transformation of a Uniform Random Variable")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
