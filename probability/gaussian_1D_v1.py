import torch
from matplotlib import pyplot as plt

# --- 1. Define Distribution Parameters ---
# Set the mean (center) of the Gaussian distribution.
mean = 0.0
# Set the standard deviation (spread) of the Gaussian distribution.
std = 1.0
# Set the total number of samples to generate.
n_samples = 100

# --- 2. Generate Samples ---
# Use the functional approach `torch.normal` to draw `n_samples` from a Gaussian
# distribution with the specified mean and standard deviation.
samples = torch.normal(mean, std, size=(n_samples,))

# --- 3. Plot the Results ---
# Create a new figure for the plot.
plt.figure(figsize=(6,4))
# Plot a histogram of the generated samples.
# - `samples.numpy()`: Converts the PyTorch tensor to a NumPy array, which is required by Matplotlib.
# - `bins=50`: Divides the data range into 50 bars.
# - `density=True`: Normalizes the histogram so that the total area of the bars equals 1.
#   This allows it to be interpreted as a probability density.
plt.hist(samples.numpy(), bins=20, density=True, alpha=0.6, color='skyblue')
plt.title(f"Gaussian Distribution (mean={mean}, std={std})")
plt.xlabel("Value")
plt.ylabel("Density")
# Display the final plot.
plt.show()