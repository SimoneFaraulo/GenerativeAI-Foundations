import torch
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define the Target Distribution p(z) ---
# This is the distribution we want to sample from. It can be unnormalized,
# meaning we only need a function proportional to the true probability density.
# We'll use a 2D Gaussian as our example target.
target_mean = torch.tensor([1.0, 1.0])
target_covariance = torch.tensor([[0.5, 0.2],
                                  [0.2, 0.8]])
# Create a PyTorch distribution object for our target. This is useful for easily
# calculating the log probability, but not strictly necessary for the algorithm itself.
p_dist = torch.distributions.MultivariateNormal(loc=target_mean, covariance_matrix=target_covariance)


# The Metropolis algorithm only requires a function that is proportional to the target density.
# We don't need to know the normalization constant.
# Here, `p_dist.log_prob(z)` gives us the log of the true density. We can exponentiate it
# to get a function proportional to p(z).
def p_unnormalized(z_val):
    # z_val will be a tensor of shape (2,) representing a point (x, y).
    return torch.exp(p_dist.log_prob(z_val))


# --- 2. Define the Proposal Distribution q(z_proposed | z_current) ---
# This distribution suggests a new candidate state `z_proposed` given the current state `z_current`.
# For the standard Metropolis algorithm, we use a symmetric proposal, where q(a|b) = q(b|a).
# A common choice is a Gaussian centered at the current state.
proposal_step_covariance = torch.tensor([[2.0, 0.0],
                                         [0.0, 2.0]])


def q_dist_generator(z_current):
    # This function returns a new proposal distribution object centered at the current state.
    return torch.distributions.MultivariateNormal(loc=z_current, covariance_matrix=proposal_step_covariance)


# --- 3. Implementazione dell'Algoritmo di Metropolis ---
# `num_samples`: The number of samples we want to collect from the target distribution.
num_samples = 50000
# `burn_in`: The number of initial samples to discard. This allows the Markov chain
# to "forget" its arbitrary starting point and converge to the stationary (target) distribution.
burn_in = 1000
samples = []

# Initialization: Choose an arbitrary starting point for the chain.
z_current = torch.tensor([-1.0, -1.0])  # We start away from the target mean.

# The main Metropolis loop. We run it for `num_samples + burn_in` iterations.
for i in range(num_samples + burn_in):
    # --- Step 1: Propose a new state ---
    # Create a proposal distribution centered at the current state.
    q_current_dist = q_dist_generator(z_current)
    # Draw a new candidate sample `z_proposed` from this distribution.
    z_proposed = q_current_dist.sample()

    # --- Step 2: Calculate the acceptance ratio (alpha) ---
    # For a symmetric proposal, the acceptance ratio simplifies to:
    # alpha = min(1, p(z_proposed) / p(z_current))
    
    # Calculate the (unnormalized) density of the target distribution at the current and proposed states.
    p_current = p_unnormalized(z_current)
    p_proposed = p_unnormalized(z_proposed)

    # Handle the edge case where we are in a region of zero probability.
    # If the current density is zero, we should always accept a move to a region
    # with potentially non-zero density.
    if p_current == 0:
        alpha = 1.0
    else:
        # Calculate the ratio. We use .item() to get a standard Python float.
        alpha = min(1.0, (p_proposed / p_current).item())

    # --- Step 3: Accept or Reject the new state ---
    # Sample a random number `u` from a Uniform(0, 1) distribution.
    u = torch.rand(1).item()
    # If `u` is less than the acceptance ratio, we accept the proposal.
    if u < alpha:
        # The new state becomes the current state for the next iteration.
        z_current = z_proposed
    # Otherwise (if u >= alpha), we reject the proposal, and the current state
    # remains the same for the next iteration. This is a crucial part of the algorithm.

    # --- Step 4: Store the sample (after the burn-in period) ---
    # We only start collecting samples after the burn-in phase is complete.
    if i >= burn_in:
        # We append a clone of the current state. .clone() is important to store a copy
        # of the tensor, not a reference to it, preventing all stored samples
        # from being overwritten with the final value of z_current.
        samples.append(z_current.clone())

# Convert the list of collected sample tensors into a single large tensor.
metropolis_samples = torch.stack(samples)

# --- 4. Visualizzazione dei Risultati ---

# Prepare a 2D grid to plot the target density p(z).
# Generate x-axis values for the plot.
# The range is centered around target_mean[0] and extends 3 standard deviations (sqrt of variance) in both directions.
x_vals = torch.linspace(target_mean[0] - 3 * target_covariance[0, 0].sqrt(),
                        target_mean[0] + 3 * target_covariance[0, 0].sqrt(), 100)
# Generate y-axis values for the plot.
# The range is centered around target_mean[1] and extends 3 standard deviations in both directions.
y_vals = torch.linspace(target_mean[1] - 3 * target_covariance[1, 1].sqrt(),
                        target_mean[1] + 3 * target_covariance[1, 1].sqrt(), 100)
# Create a 2D grid from the x and y values.
# X_grid will have x-values repeated along rows, Y_grid will have y-values repeated along columns.
# 'indexing='xy'' ensures that X_grid corresponds to columns and Y_grid to rows, which is standard for plotting.
X_grid, Y_grid = torch.meshgrid(x_vals, y_vals, indexing='xy')
# Flatten the grids and stack them to create a list of 2D points (x, y) for which to calculate the density.
# Each point in grid_points will be of the form [x_i, y_j].
grid_points = torch.stack([X_grid.flatten(), Y_grid.flatten()], dim=-1)
# Calculate the density for each point on the grid using the target distribution
# and reshape it back into a 2D grid (100x100) to match X_grid and Y_grid for plotting.
Z_density = torch.exp(p_dist.log_prob(grid_points)).reshape(100, 100)

plt.figure(figsize=(10, 8))

# Plot the target density p(z) as a filled contour plot (a heatmap).
plt.contourf(X_grid.numpy(), Y_grid.numpy(), Z_density.numpy(), levels=50, cmap='viridis', alpha=0.8)
plt.colorbar(label="Density p(z)")

# Overlay a scatter plot of the samples generated by the Metropolis algorithm.
# The samples should be concentrated in the high-density regions of p(z).
plt.scatter(metropolis_samples[:, 0].numpy(), metropolis_samples[:, 1].numpy(), s=5, alpha=0.1, color='red',
            label="Metropolis Samples")

plt.title("Metropolis Algorithm Sampling from p(z)")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
# Ensure the x and y axes have the same scale, so the circular/elliptical shape is not distorted.
plt.axis('equal')
plt.legend()
plt.grid(True)
plt.show()