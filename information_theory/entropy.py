import numpy as np
import torch
from matplotlib import pyplot as plt
import math

# --- 1. Define Distribution Parameters ---
mean = 0.0
# Try different values for std to see how entropy changes:
# std < 0.44 -> negative entropy (highly concentrated distribution)
# std > 0.44 -> positive entropy (spread-out distribution)
std = 2.0
# The number of points in the grid for numerical integration.
# A higher number leads to a more accurate result.
n_points = 4000


# --- 2. Define a function to calculate entropy ---
def calculate_entropy(mean, std, n_points):
    """
    Numerically calculates the differential entropy of a 1D Normal distribution.
    The entropy H(X) is calculated as -∫ p(x) * log2(p(x)) dx.
    """
    # Create the distribution object from the given parameters.
    norm = torch.distributions.Normal(mean, std)

    # Create a 1D grid of points for the numerical integration.
    # We use torch.linspace to keep all operations within PyTorch.
    # The range should be wide enough to cover the significant mass of the distribution.
    vals = torch.linspace(-10, 10, n_points)

    # Calculate the probability density p(x) on the grid.
    # We compute it in log-space first for better numerical stability.
    p_x_log = norm.log_prob(vals)
    p_x = torch.exp(p_x_log)

    # Calculate log2(p(x)). A small epsilon is added for numerical stability
    # to prevent log(0) = -inf if p(x) is exactly zero at the tails.
    eps = 1e-12
    log2_p_x = torch.log2(p_x.clamp(min=eps))

    # Define the integrand for the entropy calculation: -p(x) * log2(p(x)).
    integrand = -p_x * log2_p_x

    # Calculate the entropy by integrating over the grid using the trapezoidal rule.
    entropy_numeric = torch.trapz(integrand, vals).item()

    # For a Gaussian, the theoretical entropy has a closed-form solution:
    # H(X) = 0.5 * log2(2 * pi * e * sigma^2)
    entropy_theoretical = 0.5 * math.log2(2 * math.pi * math.e * (std**2))

    return entropy_numeric, entropy_theoretical

# --- 3. Calculate and Print the Results ---
numeric_h, theoretical_h = calculate_entropy(mean, std, n_points)

print(f"Distribution: Normal(mean={mean}, std={std})")
print(f"Entropy (Numeric):   {numeric_h:.4f} bits")
print(f"Entropy (Theoretical): {theoretical_h:.4f} bits")
