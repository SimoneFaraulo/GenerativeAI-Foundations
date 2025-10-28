import torch
import matplotlib.pyplot as plt

# --- 1. Define Distribution Parameters ---
# Define the mean vector for a 2D Gaussian distribution.
mean = torch.tensor([0.0, 0.0])

# Define a covariance matrix where the off-diagonal elements are non-zero.
# This non-zero covariance (-2.0) means that the variables X and Y are DEPENDENT.
cov_dep = torch.tensor([[5.0, -2.0],
                        [-2.0, 5.0]])

# Define a covariance matrix with zero off-diagonal elements.
# For a Gaussian distribution, zero covariance implies that X and Y are INDEPENDENT.
# Note: This variable is defined but not used later in the script. The main goal is to
# visualize the effect of the dependence in `cov_dep`.
cov_indep = torch.tensor([[5.0, 0.0],
                          [0.0, 5.0]])

# --- 2. Create PyTorch Distribution Objects ---
# Create the joint distribution object p(x, y) for the DEPENDENT case.
joint_dep = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov_dep)

# Create the joint distribution object for the INDEPENDENT case (for reference, though not used).
joint_indep = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov_indep)

# Create the theoretical marginal distributions, p(x) and p(y).
# For a multivariate Gaussian, the marginals are also Gaussian, and their parameters
# can be read directly from the diagonal of the covariance matrix.
marginal_x = torch.distributions.Normal(mean[0], torch.sqrt(cov_dep[0, 0]))
marginal_y = torch.distributions.Normal(mean[1], torch.sqrt(cov_dep[1, 1]))

# --- 3. Set up a 2D Grid for Evaluation ---
# Create 1D vectors of points for the x and y axes.
x = torch.linspace(-10, 10, 200)
y = torch.linspace(-10, 10, 200)

# Create 2D coordinate grids from the 1D vectors using torch.meshgrid.
# This is necessary to evaluate the 2D density function over a plane.
X, Y = torch.meshgrid(x, y, indexing="xy")

# Stack the grids to get a tensor of shape (N*N, 2), which is the required
# input format for the `log_prob` method of the MultivariateNormal distribution.
grid = torch.stack([X.flatten(), Y.flatten()], dim=-1)

# --- 4. Calculate the Densities on the Grid ---
# Calculate the true joint probability density p(x, y) for the dependent case.
# The result is reshaped back into a 200x200 grid for plotting.
Z_joint_dep = torch.exp(joint_dep.log_prob(grid)).reshape(X.shape)

# Calculate the product of the marginals, p(x) * p(y).
# If X and Y were independent, this product would be equal to the joint density p(x, y).
# The purpose of this script is to show that for the dependent case, they are NOT equal.
Z_prod = (torch.exp(marginal_x.log_prob(X)) * torch.exp(marginal_y.log_prob(Y)))

# --- 5. Plot the Results for Comparison ---
# Create a figure with two subplots side-by-side.
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: The true joint density p(x, y) for the dependent variables.
# The tilted ellipse shape visually represents the correlation between X and Y.
im1 = axs[0].contourf(X, Y, Z_joint_dep, levels=50, cmap='viridis')
axs[0].set_title("Joint density p(x,y) (Dependent)")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
plt.colorbar(im1, ax=axs[0])

# Plot 2: The product of the marginals p(x)p(y).
# This shows what the joint density would look like if the variables were independent.
# The axis-aligned shape highlights the absence of correlation.
im2 = axs[1].contourf(X, Y, Z_prod, levels=50, cmap='viridis')
axs[1].set_title("Product of marginals p(x)p(y) (If independent)")
axs[1].set_xlabel("X")
axs[1].set_ylabel("Y")
plt.colorbar(im2, ax=axs[1])

# Adjust layout to prevent titles from overlapping and display the plot.
plt.tight_layout()
plt.show()
