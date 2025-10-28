import torch
from matplotlib import pyplot as plt

# --- 1. Define Distribution Parameters ---
mean_z = 0.0
mean_q = 1.0
std_z = 1.0
std_q = 1.0
num_monte_carlo_samples = 100 # Number of samples for Monte Carlo estimation
num_grid_points = 1000 # Number of points for numerical integration grid

# --- 2. Create Distribution Objects ---
norm_z = torch.distributions.Normal(mean_z, std_z) # Corrected std_z instead of mean_q
norm_q = torch.distributions.Normal(mean_q, std_q)

# Create a grid of values for numerical integration and plotting.
vals_grid = torch.linspace(-5, 5, num_grid_points) # Adjusted range for better visualization of both distributions

# --- 3. Define the Function f(z) and Densities on the Grid ---
# The function f(z) for which we want to calculate the expected value E_p[f(Z)].
f_z_grid = torch.tanh(vals_grid)

# Calculate the PDF values for p(z) and q(z) on the grid.
p_z_grid = torch.exp(norm_z.log_prob(vals_grid))
q_z_grid = torch.exp(norm_q.log_prob(vals_grid))

# --- 4. Calculate Expected Value E_p[f(Z)] via Direct Numerical Integration ---
# This is the "ground truth" value we want to estimate.
# E_p[f(Z)] = integral[ f(z) * p(z) dz ]
expected_f_z_numeric = torch.trapz(p_z_grid * f_z_grid, vals_grid).item()

# --- 5. Implement Importance Sampling (Monte Carlo Estimation) ---

# Generate samples from the PROPOSAL distribution q(z).
# This is the core of Monte Carlo sampling.
z_samples_from_q = norm_q.sample((num_monte_carlo_samples,))

# Evaluate the function f(z) at these samples.
f_z_samples = torch.tanh(z_samples_from_q)

# Calculate the PDF values for p(z) and q(z) at the sampled points.
p_z_at_samples = torch.exp(norm_z.log_prob(z_samples_from_q))
q_z_at_samples = torch.exp(norm_q.log_prob(z_samples_from_q))

# Calculate the importance weights for each sample.
# Add a small epsilon to the denominator to prevent division by zero if q(z) is extremely small.
epsilon = 1e-10 # Small constant for numerical stability
importance_weights_samples = p_z_at_samples / (q_z_at_samples + epsilon)

# Estimate E_p[f(Z)] using the importance sampling formula.
# E_p[f(Z)] approx (1/N) * sum[ f(z_i) * w_i ]
expected_f_z_importance_sampling = torch.mean(f_z_samples * importance_weights_samples).item()

# --- 6. Print Results ---
print(f"Target Distribution p(z): N(mean={mean_z}, std={std_z})")
print(f"Proposal Distribution q(z): N(mean={mean_q}, std={std_q})")
print(f"Function f(z) = tanh(z)")
print(f"Number of Monte Carlo samples: {num_monte_carlo_samples}")
print("-" * 40)
print(f"E_p[f(Z)] (Direct Numerical Integration): {expected_f_z_numeric:.6f}")
print(f"E_p[f(Z)] (Importance Sampling Estimate): {expected_f_z_importance_sampling:.6f}")

# --- 7. Plotting for Visualization ---
plt.figure(figsize=(12, 7))

# --- Scatter plot of the weighted samples ---
# This is the core visualization of what importance sampling does.
# Each point shows the contribution of a single sample z_i drawn from q(z).
# The y-value is f(z_i) * w_i, where w_i is the importance weight p(z_i)/q(z_i).
# We expect to see high values where p(z) is large but q(z) is small.
weighted_f_z_samples = f_z_samples * importance_weights_samples
mask = (weighted_f_z_samples <= 1) & (weighted_f_z_samples >= -1)
weighted_f_z_samples = weighted_f_z_samples[mask]
z_samples_from_q = z_samples_from_q[mask]

plt.scatter(z_samples_from_q.numpy(), weighted_f_z_samples.numpy(), label='Weighted Samples f(z_i)w_i', color='orange', linewidths=2.5,alpha=0.2, s=15, zorder=3)

# Plot the target distribution p(z)
plt.plot(vals_grid.numpy(), p_z_grid.numpy(), label='Target p(z)', color='blue', linewidth=2.5, alpha=0.8, zorder=2)
# Plot the proposal distribution q(z)
plt.plot(vals_grid.numpy(), q_z_grid.numpy(), label='Proposal q(z)', color='red', linestyle='--', linewidth=2.5, alpha=0.8, zorder=2)
# Plot the function f(z)
plt.plot(vals_grid.numpy(), f_z_grid.numpy(), label='Function f(z) = tanh(z)', color='green', linestyle=':', linewidth=2.5, zorder=2)
# Plot the integrand f(z)*p(z)
plt.plot(vals_grid.numpy(), (f_z_grid * p_z_grid).numpy(), label='True Integrand f(z)p(z)', color='purple', linestyle='-', alpha=0.7, zorder=1)

plt.legend()
plt.title("Importance Sampling Visualization")
plt.xlabel("z")
plt.ylabel("Density / Value")
plt.grid(True)
plt.show()
