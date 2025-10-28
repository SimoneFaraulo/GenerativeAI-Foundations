import torch
import matplotlib.pyplot as plt
from math import pi


# --- Parameters for Rejection Sampling ---
num_desired_samples = 10000 # The number of samples we want to obtain from the target distribution.

# --- 1. Define the Target Distribution p(z) ---
# For rejection sampling, p(z) must be non-negative.
# We'll use sin(z) on the interval [0, pi], where it is positive.
z_grid = torch.linspace(0, pi, 1000) # Grid for evaluating p(z) and q(z) for plotting and k calculation
p_z_target = torch.sin(z_grid) # Target PDF (unnormalized)

# --- 2. Define the Proposal Distribution q(z) ---
# A Normal distribution centered at pi/2, covering the [0, pi] range.
mean_q_proposal = pi / 2
std_q_proposal = pi / 4 # Adjust std to cover the range well (e.g., 2*std covers ~95% of mass)
norm_proposal = torch.distributions.Normal(mean_q_proposal, std_q_proposal)
q_z_proposal = torch.exp(norm_proposal.log_prob(z_grid))

# --- 3. Calculate the constant k ---
# k must satisfy p_z_target(z) <= k * q_z_proposal(z) for all z.
# This means k = max(p_z_target(z) / q_z_proposal(z)).
# We add a small buffer (e.g., 5%) to k to account for numerical inaccuracies and ensure the envelope holds.
ratio = p_z_target / q_z_proposal
k = torch.max(ratio).item() * 1.05 # Get the scalar value and add a buffer

print(f"Calculated k: {k:.4f}")

# --- 4. Rejection Sampling Loop ---
accepted_samples = []
num_accepted = 0
num_generated_proposals = 0

while num_accepted < num_desired_samples:
    # 1. Sample z_star from the proposal distribution q(z)
    z_star = norm_proposal.sample()
    num_generated_proposals += 1

    # 2. Check if z_star is within the domain where p(z) is defined and non-zero ([0, pi]).
    # Our p_z_target is sin(z) on [0, pi], and 0 elsewhere.
    # If z_star is outside this range, p_z_target(z_star) is effectively 0, so it will be rejected.
    if z_star >= 0 and z_star <= pi:
        # Evaluate p_z_target(z_star) and q_z_proposal(z_star) at the sampled point.
        p_at_z_star = torch.sin(z_star)
        q_at_z_star = torch.exp(norm_proposal.log_prob(z_star))

        # 3. Sample u from Uniform(0, k * q_z_proposal(z_star))
        # The upper bound for the uniform distribution is k * q_z_proposal(z_star).
        # Add a small epsilon to q_at_z_star to prevent issues if it's exactly zero (though unlikely for Normal).
        if q_at_z_star > 1e-9:
            u = torch.distributions.Uniform(0.0, k * q_at_z_star).sample()

            # 4. Acceptance condition: u < p_z_target(z_star)
            if u < p_at_z_star:
                accepted_samples.append(z_star)
                num_accepted += 1
    # If z_star is outside [0, pi], p_z_target(z_star) is 0, so it's implicitly rejected.

print(f"Total proposals generated: {num_generated_proposals}")
print(f"Total samples accepted: {num_accepted}")
print(f"Efficiency: {num_accepted / num_generated_proposals:.2%}")

# Convert the list of accepted samples to a tensor for plotting.
final_samples = torch.stack(accepted_samples)

# --- 5. Plotting the Results ---
plt.figure(figsize=(12, 7))

# Plot the target distribution p(z)
plt.plot(z_grid.numpy(), p_z_target.numpy(), label='Target p(z) = sin(z)', color='blue', linewidth=2)

# Plot the scaled proposal distribution k * q(z) (the envelope)
plt.plot(z_grid.numpy(), (k * q_z_proposal).numpy(), label='Scaled Proposal k * q(z)', color='red', linestyle='--', linewidth=1)

# Plot a histogram of the generated samples
plt.hist(final_samples.numpy(), bins=50, density=True, alpha=0.6, color='green', label='Generated Samples (Rejection Sampling)')

plt.title("Rejection Sampling for p(z) = sin(z) on [0, pi]")
plt.xlabel("z")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()
