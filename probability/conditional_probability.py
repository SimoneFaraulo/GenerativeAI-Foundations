import torch
import matplotlib.pyplot as plt

# --- 1. Define the Joint Distribution Parameters ---
# We start by defining a 2D multivariate normal (Gaussian) distribution.
mean = torch.tensor([0.0, 0.0])
cov = torch.tensor([[5.0, 0.0],
                    [0.0, 5.0]])

# Create the joint distribution object p(X, Y).
joint = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)

# --- 2. Define the Theoretical Marginal Distributions ---
# For a multivariate Gaussian, the marginal distributions are also Gaussian.
# We can extract their parameters directly from the joint mean and covariance matrix.
mean_x = mean[0]
mean_y = mean[1]
sigma_xx = cov[0, 0]
sigma_yy = cov[1, 1]
sigma_xy = cov[0, 1]   # Covariance between X and Y

# Create the marginal distribution object for Y, p(Y).
marginal_y = torch.distributions.Normal(mean_y, torch.sqrt(sigma_yy))
# Create the marginal distribution object for X, p(X).
marginal_x = torch.distributions.Normal(mean_x, torch.sqrt(sigma_xx))

# --- 3. Setup the Grid and Conditioning Values ---
# Create a 1D grid of x-values. This will be used for numerical calculations and plotting.
x_vals = torch.linspace(-10, 10, 400)

# Choose the specific value(s) of Y on which we want to condition.
# The code is structured to loop over multiple values, but here we use just one.
y0_list = torch.tensor([-2.0])

# --- 4. Plotting Setup ---
plt.figure(figsize=(10, 6))

# --- 5. Main Loop: Calculate and Analyze for each y0 ---
# Iterate over each value y0 we want to condition on.
for i, y0 in enumerate(y0_list):
    # --- 5.1. Numerically compute the conditional PDF p(x|y0) ---
    # We use the fundamental formula: p(x|y0) = p(x, y0) / p(y0).
    # To evaluate the joint p(x, y0) for all x in x_vals, we need to create pairs (x_i, y0).
    
    # First, create a tensor of y0 values with the same length as x_vals.
    y0_expand = y0.expand(x_vals.shape[0])
    # Stack x_vals and y0_expand to create a tensor of shape (N, 2) of points [(x0, y0), (x1, y0), ...].
    points = torch.stack([x_vals, y0_expand], dim=1)
    # Calculate the log of the joint probability log[p(x_i, y0)] for each point. Using log-space is numerically more stable.
    joint_logp = joint.log_prob(points)
    # Convert from log-probability to probability to get the joint density p(x_i, y0).
    joint_p = torch.exp(joint_logp)

    # Calculate the value of the marginal p(y0), which is the denominator in our formula.
    p_y0 = torch.exp(marginal_y.log_prob(y0))
    # Finally, compute the conditional PDF by dividing the joint by the marginal.
    pdf_numeric = joint_p / p_y0

    # --- 5.2. Compute the marginal p(x) for comparison ---
    # We calculate the theoretical marginal p(x) on the same grid to compare it with the conditional p(x|y0).
    p_x = torch.exp(marginal_x.log_prob(x_vals))

    # --- 5.3. Plot the distributions ---
    # Plot the numerically computed conditional distribution p(x|y0).
    plt.plot(x_vals.numpy(), pdf_numeric.numpy(), '-', label=f"p(x|y={y0.item():.1f})", linewidth=1)
    # Plot the theoretical marginal distribution p(x) for comparison.
    plt.plot(x_vals.numpy(), p_x.numpy(), '--', label="p(x) (marginal)", linewidth=1)

    # --- 5.4. Perform Quantitative Checks ---
    
    # Check 1: Maximum absolute difference between the two PDFs.
    # This measures the largest pointwise discrepancy.
    max_abs_diff = torch.max(torch.abs(pdf_numeric - p_x)).item()
    
    # Check 2: L1 difference, which is the integral of the absolute difference.
    # This represents the total "area" of difference between the two curves.
    l1_diff = torch.trapz(torch.abs(pdf_numeric - p_x), x_vals).item()
    
    # Check 3: KL Divergence D_KL( p(x|y) || p(x) )
    # This measures how much information is lost when p(x) is used to approximate p(x|y).
    # A small epsilon is added to prevent log(0) which would result in -inf.
    eps = 1e-12 # A small constant for numerical stability.
    q = pdf_numeric.clamp(min=eps)  # The conditional PDF, p(x|y).
    p = p_x.clamp(min=eps)          # The marginal PDF, p(x).
    # Numerically integrate the KL divergence formula: integral[ q(x) * log(q(x)/p(x)) dx ].
    kl = torch.trapz(q * (torch.log(q) - torch.log(p)), x_vals).item()

    # Check 4: Expected value E[X | Y=y0] (Conditional Mean)
    # This is the mean of the conditional distribution, calculated as integral[ x * p(x|y0) dx ].
    conditional_mean = torch.trapz(x_vals * pdf_numeric, x_vals).item()

    # Check 5: Sanity check for p(y0).
    # We can recover the marginal p(y0) by integrating the joint p(x, y0) over all x.
    # The result should be very close to the value from the theoretical marginal formula.
    p_y0_from_integration = torch.trapz(joint_p, x_vals).item()

    # Check 6: Mode of the conditional distribution p(x|y0).
    # The mode is the value of x where the PDF reaches its maximum.
    # For a Gaussian, the mode, mean, and median should all coincide.
    max_pdf_index = torch.argmax(pdf_numeric)
    conditional_mode = x_vals[max_pdf_index].item()

    # --- 5.5. Add annotations to the plot ---
    # Add vertical lines to mark the calculated mean and mode of the conditional distribution.
    plt.axvline(x=conditional_mean, color='g', linestyle=':', linewidth=2, label=f"Mean of p(x|y)={y0.item():.1f}")
    plt.axvline(x=conditional_mode, color='r', linestyle=':', linewidth=2, label=f"Mode of p(x|y={y0.item():.1f})")

    # --- 5.6. Print the results of the quantitative checks ---
    print(f"--- y0 = {y0.item():.2f} ---")
    print(f"Max abs diff sup_x |p(x|y)-p(x)| = {max_abs_diff:.6f}")
    print(f"L1 diff (integral |p(x|y)-p(x)| dx) = {l1_diff:.6f}")
    print(f"KL( p(x|y) || p(x) ) approx = {kl:.6f}")
    print(f"E[X | Y=y0] (conditional mean) = {conditional_mean:.6f}")
    print(f"Mode of p(x|y0) (numeric) = {conditional_mode:.6f}")
    print(f"p(y0) (from marginal formula) = {p_y0.item():.6f}")
    print(f"p(y0) (from integrating joint) = {p_y0_from_integration:.6f}")
    print()

# --- 6. Finalize and Show Plot ---
plt.xlabel("x")
plt.ylabel("Density")
plt.title("Comparison: conditional p(x|y0) (solid) vs marginal p(x) (dashed)")
plt.legend()
plt.grid(True)
plt.show()