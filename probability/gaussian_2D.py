import torch
import matplotlib.pyplot as plt

# --- 1. Define the 2D Distribution Parameters ---
# Define the mean vector [E[X], E[Y]].
mean = torch.tensor([0.0, 0.0])
# Define the covariance matrix. The non-zero off-diagonal elements (-2.0)
# indicate that the variables X and Y are correlated (and thus, not independent).
cov = torch.tensor([[5.0, -2.0],
                    [-2.0, 5.0]])

# --- 2. Create the Distribution Object ---
# Create a MultivariateNormal distribution object using the specified mean and covariance.
dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)

# --- 3. Generate Samples from the Distribution ---
# Draw 5000 random samples. Each sample is a 2D point (x, y).
# The resulting tensor `samples` will have a shape of (5000, 2).
samples = dist.sample((5000,))

# --- 4. Set up a Grid for Density Evaluation ---
# Create 1D vectors of points for the x and y axes. This grid will be used
# to visualize the PDF and perform numerical integration.
x = torch.linspace(-4, 4, 100)
y = torch.linspace(-4, 4, 100)
# Create 2D coordinate grids from the 1D vectors using torch.meshgrid.
X, Y = torch.meshgrid(x, y, indexing='xy')

# Stack the 2D grids to get a tensor of shape (100*100, 2). This is the
# required input format for the `log_prob` method.
grid = torch.stack([X.flatten(), Y.flatten()], dim=-1)

# --- 5. Calculate the Joint PDF on the Grid ---
# Calculate the log-probability of the joint distribution at each point in the grid.
# Then, use torch.exp() to get the actual probability density values.
Z = torch.exp(dist.log_prob(grid))
# Reshape the flat density tensor back into a 100x100 grid for plotting.
Z = Z.reshape(100, 100)

# --- 6. Compute Marginals via Numerical Integration ---
# Calculate the marginal density p(x) by integrating the joint density p(x, y) over y.
# `torch.trapz(Z, y, dim=0)` performs numerical integration along dimension 0 (the y-axis).
p_x = torch.trapz(Z, y, dim=0)
# Calculate the marginal density p(y) by integrating the joint density p(x, y) over x.
# `torch.trapz(Z, x, dim=1)` performs numerical integration along dimension 1 (the x-axis).
p_y = torch.trapz(Z, x, dim=1)

# --- 7. Plot the Results ---
# Create a figure with 3 subplots arranged vertically.
plt.figure(figsize=(6, 10))

# Subplot 1: A contour plot (heatmap) of the joint density p(x, y).
plt.subplot(3,1,1)
plt.contourf(X.numpy(), Y.numpy(), Z.numpy(), levels=50, cmap='viridis')
plt.colorbar(label="Density")
plt.title("2D Gaussian Density Heatmap")
plt.xlabel("X")
plt.ylabel("Y")
plt.axis('equal') # Ensure the scaling is the same on both axes.

# Subplot 2: The calculated marginal density p(x).
plt.subplot(3,1,2)
plt.plot(x, p_x)
plt.title("Marginal p(x) via integration")
plt.xlabel("x")
plt.ylabel("Density")

# Subplot 3: The calculated marginal density p(y).
plt.subplot(3,1,3)
plt.plot(y, p_y)
plt.title("Marginal p(y) via integration")
plt.xlabel("y")
plt.ylabel("Density")

# Adjust layout to prevent titles and labels from overlapping.
plt.tight_layout()
# Display the final plot.
plt.show()
