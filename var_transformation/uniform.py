import numpy as np
import torch
from matplotlib import pyplot as plt

# --- 1. Define the original distribution p(z) ---
# We start by defining a uniform distribution for our original random variable 'z'.
# It is defined over the interval [0, 1].
n_points = 1000
z_range_low = 0.0
z_range_high = 1.0

# Create a wide grid of points. This will serve as the x-axis for plotting
# and allow us to see where the densities are zero.
z_vals = torch.linspace(-2, 12, n_points)

# --- Manually calculate the probability density p(z) ---
# The density of a Uniform(a, b) is 1/(b-a) inside the interval [a, b] and 0 otherwise.
# This avoids the `ValueError` from `log_prob` for points outside the support.

# Start with a tensor of zeros for the density.
p_z = torch.zeros_like(z_vals)
# Create a boolean mask to find which points in our grid fall within the [0, 1] interval.
mask_z = (z_vals >= z_range_low) & (z_vals <= z_range_high)
# For the points inside the interval, set the density to its correct value.
p_z[mask_z] = 1.0 / (z_range_high - z_range_low)

# --- 2. Define the transformation y = g(z) ---
# We define a linear transformation to map z ~ U(0, 1) to a new variable y ~ U(a, b).
# The correct formula for this mapping is: y = (b - a) * z + a
a, b = 3.0, 7.0
transformation = lambda z: (b - a) * z + a

# Apply the transformation to the grid of z values to get the new y values.
# This is the crucial step that maps the original value axis to the new, transformed axis.
y_vals = transformation(z_vals)

# --- 3. Calculate the new density p(y) ---
# According to the change of variables formula for a linear transformation:
# p_y(y) = p_z(g_inverse(y)) * |1/a|
#
# For our transformation, the Jacobian is (b-a), so the new density height is 1 / (b-a).
# The new support for y is the interval [a, b].
y_range_low, y_range_high = a, b

# Manually define the density for p(y) over its new support.
p_y = torch.zeros_like(y_vals)
# Create a mask for the points that fall within the new support [a, b].
mask_y = (y_vals >= y_range_low) & (y_vals <= y_range_high)
# The height of the new uniform distribution is 1 / (new_width).
p_y[mask_y] = 1.0 / (y_range_high - y_range_low)

# --- 4. Plot the results for comparison ---
plt.figure(figsize=(10, 6))
# Plot p(z) against its own axis, z_vals.
plt.plot(z_vals.numpy(), p_z.numpy(), label='Original density p(z) ~ U(0,1)', color='blue', linewidth=2)
# Plot p(y) against ITS OWN TRANSFORMED axis, y_vals.
plt.plot(y_vals.numpy(), p_y.numpy(), label=f'Transformed density p(y) ~ U({a:.1f}, {b:.1f})', color='red', linestyle='--', linewidth=2)

plt.title("Transformation of a Uniform Random Variable")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()