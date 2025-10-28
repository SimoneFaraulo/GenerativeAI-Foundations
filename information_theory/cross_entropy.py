import numpy as np
import torch
from matplotlib import pyplot as plt
from entropy import calculate_entropy

# --- 1. Define Distribution Parameters ---
# Define the parameters for two normal distributions, X and Y.
mean_x = 0.0
std_x = 2.0
mean_y = 1.0
std_y = 1.0
# The number of points in the grid for numerical integration.
# A higher number leads to a more accurate result.
n_points = 4000


# --- 2. Define a function to calculate cross-entropy ---
def calculate_cross_entropy(mean_x, std_x, mean_y, std_y, n_points):
    """
    Numerically calculates the cross-entropy between two 1D Normal distributions.
    The cross-entropy H(X, Y) is calculated as -∫ p(x) * log2(q(x)) dx,
    where p is the distribution of X and q is the distribution of Y.
    """
    # Create the distribution objects from the given parameters.
    norm_x = torch.distributions.Normal(mean_x, std_x)
    norm_y = torch.distributions.Normal(mean_y, std_y)

    # Create a 1D grid of points for the numerical integration.
    # The range should be wide enough to cover the significant mass of both distributions.
    vals = torch.linspace(-10, 10, n_points)

    # Calculate the probability density p(x) on the grid for distribution X.
    p_x_log = norm_x.log_prob(vals)
    p_x = torch.exp(p_x_log)

    # Calculate the probability density q(y) on the grid for distribution Y.
    p_y_log = norm_y.log_prob(vals)
    p_y = torch.exp(p_y_log)

    # Calculate log2(q(y)). A small epsilon is added for numerical stability
    # to prevent log(0) = -inf if q(y) is exactly zero at the tails.
    eps = 1e-12
    log2_p_y = torch.log2(p_y.clamp(min=eps))

    # Define the integrand for the cross-entropy calculation: -p(x) * log2(q(y)).
    integrand = -p_x * log2_p_y

    # Calculate the cross-entropy by integrating over the grid using the trapezoidal rule.
    cross_entropy_numeric = torch.trapz(integrand, vals).item()

    return cross_entropy_numeric, p_x, p_y

# --- 3. Calculate and Print the Results ---
# Calculate the entropy of distribution X.
numeric_h_x, _ = calculate_entropy(mean_x, std_x, n_points)
# Calculate the cross-entropy between X and Y.
numeric_ce_xy, p_x, p_y = calculate_cross_entropy(mean_x, std_x, mean_y, std_y, n_points)

# Calculate the entropy of distribution Y.
numeric_h_y, _ = calculate_entropy(mean_y, std_y, n_points)
# Calculate the cross-entropy between Y and X.
numeric_ce_yx, _, _= calculate_cross_entropy(mean_y, std_y, mean_x, std_x, n_points)

# Create the x-axis values for plotting.
vals = np.linspace(-10,10, n_points)

# Calculate the Kullback-Leibler (KL) divergence.
# KL(X || Y) = H(X, Y) - H(X)
kl_divergence_xy = numeric_ce_xy - numeric_h_x
# KL(Y || X) = H(Y, X) - H(Y)
kl_divergence_yx = numeric_ce_yx - numeric_h_y


print(f"Distribution p(x): Normal(mean={mean_x}, std={std_x})")
print(f"Distribution q(y): Normal(mean={mean_y}, std={std_y})")
print(f"Cross Entropy (Numeric) H(X, Y):   {numeric_ce_xy:.4f} bits")
print(f"Kullback-Leibler Divergence KL(X || Y): {kl_divergence_xy:.4f} bits")
print(f"Cross Entropy (Numeric) H(Y, X):   {numeric_ce_yx:.4f} bits")
print(f"Kullback-Leibler Divergence KL(Y || X): {kl_divergence_yx:.4f} bits")


# --- 4. Plot the Distributions ---
plt.figure(figsize=(10, 6))
plt.plot(vals, p_x, '-', label = "p(x)", color='green')
plt.plot(vals, p_y, '-', label = "q(y)", color='red')
plt.title("Probability Distributions p(x) and q(y)")
plt.xlabel("Value")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()
