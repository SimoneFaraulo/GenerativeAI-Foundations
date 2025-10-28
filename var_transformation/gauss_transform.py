import matplotlib.pyplot as plt
import numpy as np
import torch

# --- 1. Define the original distribution p(z) ---
# We define a Gaussian distribution for our original variable 'z'.
mean_z = 5.0
std_z = 2.0
z_range = np.array([-10, 10])
n_points = 1000 # Number of points for the numerical grid.

# Create the Normal distribution object.
norm_z = torch.distributions.Normal(mean_z, std_z)
# Create a grid of values for the z-axis.
z_vals = torch.from_numpy(np.linspace(z_range[0], z_range[1], n_points)).float()
# Calculate the density p(z) on its grid.
p_z = torch.exp(norm_z.log_prob(z_vals))

# --- 2. Define the transformation y = g(z) ---
# We define a standardization transformation: y = (z - mean_z) / std_z
# This should transform our N(mean_z, std_z) distribution into a standard normal N(0, 1).
y_vals = (z_vals - mean_z) / std_z

# --- 3. Calculate the new density p(y) using the Change of Variables formula ---
# The formula is: p_y(y) = p_z(g_inverse(y)) * |Jacobian|
#
# 1. Transformation g(z): y = (z - mean_z) / std_z
# 2. Inverse transformation g_inverse(y): z = (y * std_z) + mean_z
# 3. Jacobian (derivative of the inverse w.r.t y): d(z)/dy = std_z

# We can calculate the Jacobian explicitly using autograd for demonstration.
# First, create a tensor for y that requires gradient tracking.
y_for_grad = y_vals.clone().requires_grad_()
# Define the inverse function z = g_inverse(y).
z_from_y = (y_for_grad * std_z) + mean_z
# Use autograd to compute the derivative of z_from_y with respect to y_for_grad.
# The result should be a constant tensor of value std_z (which is 2.0).
jacobian = torch.autograd.grad(z_from_y, y_for_grad, torch.ones_like(y_for_grad))[0]

# Now, apply the formula correctly:
# First, evaluate the original density p_z at the inverse-transformed points g_inverse(y).
# This corresponds to `(y_for_grad * std_z) + mean_z`.
p_z_at_g_inv = torch.exp(norm_z.log_prob((y_for_grad * std_z) + mean_z))

# Then, multiply by the absolute value of the Jacobian.
p_y = p_z_at_g_inv * torch.abs(jacobian)

# --- 4. Plot the results ---
# The new distribution y = (z - mean_z) / std_z should be a standard normal N(0, 1).
plt.figure(figsize=(10, 6))
# Plot the original distribution p(z) against its own axis z_vals.
plt.plot(z_vals.numpy(), p_z.numpy(), label=f'p(z) ~ N(mean={mean_z}, std={std_z})')
# Plot the new distribution p(y) against its transformed axis y_vals.
# We use .detach() to remove the gradient history from p_y before converting to numpy.
new_mean = (mean_z - mean_z) / std_z
plt.plot(y_vals.numpy(), p_y.detach().numpy(), label=f'p(y) ~ N(mean={new_mean:.1f}, std=1.0)', linestyle='--')
plt.title("Transformation of a Gaussian Variable")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()