import numpy as np
import torch
from matplotlib import pyplot as plt

# --- 1. Define Parameters for a 2D Gaussian Distribution ---

# Define the parameters for the marginal distribution of X.
mean_x = torch.tensor(.0)
std_x = torch.tensor(5.0)

# Define the parameters for the marginal distribution of Y.
mean_y = torch.tensor(2.0)
std_y = torch.tensor(1.0)

# Define the theoretical covariance between X and Y.
cov = 2.0

# --- 2. Construct the Joint Distribution ---

# The mean of the joint distribution is a vector of the individual means.
joint_mean = torch.stack([mean_x, mean_y], dim=-1)
print(f"Joint mean: {joint_mean}")

# The covariance matrix is constructed from the variances (std_dev^2) and the covariance.
# The diagonal contains the variances Var(X) and Var(Y).
# The off-diagonal contains the covariance Cov(X, Y).
joint_cov = torch.tensor([[std_x**2, cov],
                          [cov, std_y**2]])

# --- 3. Create PyTorch Distribution Objects ---

# Create distribution objects for the theoretical marginals p(x) and p(y).
x = torch.distributions.Normal(mean_x, std_x)
y = torch.distributions.Normal(mean_y, std_y)

# Create the main distribution object for the joint probability p(x, y).
joint = torch.distributions.MultivariateNormal(joint_mean, covariance_matrix=joint_cov)

# --- 4. Set up Grids for Numerical Computation ---

# Create 1D grids of values for x and y. These are the points where we will evaluate the densities.
# We convert them to float32 tensors, which is the standard for PyTorch operations.
x_vals = torch.from_numpy(np.linspace(-20, 20, 400)).float()
y_vals = torch.from_numpy(np.linspace(-20, 20, 400)).float()

# Create 2D coordinate grids from the 1D vectors.
# `x_grid` and `y_grid` will be 400x400 matrices.
# `x_grid[i, j]` and `y_grid[i, j]` give the (x, y) coordinates of a point on the plane.
x_grid, y_grid = torch.meshgrid(x_vals, y_vals, indexing='xy')

# Stack the 2D grids to create a tensor of shape (400, 400, 2).
# This format is required by `joint.log_prob` to evaluate the density at each (x, y) point.
joint_points = torch.stack([x_grid , y_grid], dim=-1)

# --- 5. Calculate Densities on the Grids ---

# Calculate the marginal density p(x) for each value in x_vals.
p_x = torch.exp(x.log_prob(x_vals))

# Calculate the marginal density p(y) for each value in y_vals.
p_y = torch.exp(y.log_prob(y_vals))

# Calculate the joint density p(x, y) for each point on the 2D grid.
# The result is reshaped to match the 400x400 grid structure.
p_joint = torch.exp(joint.log_prob(joint_points))

# --- 6. Numerically Calculate Moments of the Distributions ---

# Calculate the expected value E[X] by numerically integrating ∫ x*p(x) dx.
exp_val_x = torch.trapz(p_x*x_vals, x_vals).item()
print(f"E[X] = {exp_val_x}")

# Calculate the expected value E[Y] by numerically integrating ∫ y*p(y) dy.
exp_val_y = torch.trapz(p_y*y_vals, y_vals).item()
print(f"E[Y] = {exp_val_y}")

# Calculate the variance of X using the formula Var(X) = E[X^2] - (E[X])^2.
e_x2 = torch.trapz(p_x*(x_vals*x_vals), x_vals).item()
var_x_numeric = e_x2 - exp_val_x**2
std_x_numeric = np.sqrt(var_x_numeric)
print(f"Var[X] (numeric) = {var_x_numeric:.4f} (Teorica: {std_x**2:.4f})")

# Calculate the variance of Y using the formula Var(Y) = E[Y^2] - (E[Y])^2.
e_y2 = torch.trapz(p_y*(y_vals*y_vals), y_vals).item()
var_y_numeric = e_y2 - exp_val_y**2
std_y_numeric = np.sqrt(var_y_numeric)
print(f"Var[Y] (numeric) = {var_y_numeric:.4f} (Teorica: {std_y**2:.4f})")

# Calculate the covariance using the formula Cov(X, Y) = E[XY] - E[X]E[Y].
# First, calculate E[XY] via a double integral: ∫∫ x*y*p(x,y) dx dy.
integrand_xy = p_joint * x_grid * y_grid
integrand_over_y = torch.trapz(integrand_xy, y_vals, dim=0) # Integrate over y first
e_xy = torch.trapz(integrand_over_y, x_vals, dim=0).item()
cov_xy_numeric = e_xy - exp_val_x*exp_val_y
print(f"Cov[X,Y] (numeric) = {cov_xy_numeric:.4f} (Teorica: {cov:.4f})")

# --- 7. Plot the Results ---

# Create a new figure for the plot.
plt.figure(figsize=(10, 6))

# Draw a horizontal line at y=0 to represent the x-axis.
plt.axhline(y=0, color='black', linestyle='--', linewidth=0.8)

# Draw vertical dotted lines to mark the means (expected values) of X and Y.
plt.axvline(x=exp_val_x, color='g', linestyle=':', linewidth=2, label=f'E[X] = {exp_val_x:.2f}')
plt.axvline(x=exp_val_y, color='r', linestyle=':', linewidth=2, label=f'E[Y] = {exp_val_y:.2f}')

# Draw horizontal lines to visualize the standard deviation (mean ± std_dev) for each distribution.
# These are placed at arbitrary y-positions for visibility.
plt.hlines(y=-0.005, xmin=exp_val_x-std_x_numeric, xmax=exp_val_x+std_x_numeric, color='g', linewidth=2.5, label=f'StdDev(X) ≈ {std_x_numeric:.2f}')
plt.hlines(y=-0.010, xmin=exp_val_y-std_y_numeric, xmax=exp_val_y+std_y_numeric, color='r', linewidth=2.5, label=f'StdDev(Y) ≈ {std_y_numeric:.2f}')

# Plot the marginal probability density functions for X and Y.
plt.plot(x_vals, p_x, '-', label = "p(x)", color='green')
plt.plot(y_vals, p_y, '-', label = "p(y)", color='red')

# Set plot labels, title, legend, and grid.
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Marginal Densities p(x) and p(y)")
plt.legend()
plt.grid(True)

# Display the final plot.
plt.show()