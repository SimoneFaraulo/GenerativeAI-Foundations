import torch
import matplotlib.pyplot as plt

# --- 1. Define and Create the Distribution Object ---
# Set the mean and standard deviation for the distribution.
mean = 0.0
std = 1.0
# Create a Normal distribution object from the `torch.distributions` module.
# This object encapsulates the properties and methods of the distribution.
dist = torch.distributions.Normal(mean, std)

# --- 2. Generate Samples from the Distribution ---
# Use the `.sample()` method of the distribution object to draw 10,000 samples.
samples = dist.sample((10000,))

# --- 3. Calculate the Theoretical Probability Density Function (PDF) ---
# Create a 1D grid of x-values where we will evaluate the PDF.
x = torch.linspace(-4, 4, 200)
# Calculate the log-probability of each point in `x` using `.log_prob()`.
# Working in log-space is generally more numerically stable than direct probability.
# Then, use `torch.exp()` to convert the log-probabilities back to actual probability densities.
pdf = torch.exp(dist.log_prob(x))

# --- 4. Plot Both Samples and the Theoretical PDF ---
# Create a new figure for the plot.
plt.figure(figsize=(6,4))
# Plot a normalized histogram of the generated samples. This shows the empirical distribution.
plt.hist(samples.numpy(), bins=50, density=True, alpha=0.5, label='Samples')
# Overlay the theoretical PDF curve on the same plot.
# This shows how well the distribution of the samples matches the true mathematical form.
plt.plot(x.numpy(), pdf.numpy(), 'r-', label='PDF')
plt.title("Normal Distribution with PyTorch")
plt.xlabel("x")
plt.ylabel("Density")
# Add a legend to identify the histogram and the PDF curve.
plt.legend()
# Display the final plot.
plt.show()
