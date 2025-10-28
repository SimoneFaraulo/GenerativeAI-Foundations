import torch
import matplotlib.pyplot as plt
# Import the custom UniformDisk distribution class from the other file.
from circle_distribution import UniformDisk

# --- 1. Use Our Custom Distribution ---

# Define the parameters for our sampling process.
circle_radius = 1.0
num_samples = 5000

# Create an instance of our custom distribution, just like we would for a built-in one.
disk_dist = UniformDisk(radius=circle_radius)

# Generate the data by calling the `.sample()` method.
# This will execute the inverse transform sampling logic we defined inside the class.
samples = disk_dist.sample(sample_shape=(num_samples,))

# The `samples` tensor has a shape of (num_samples, 2).
# We extract the x and y coordinates for separate analysis.
# `[:, 0]` selects all rows and the first column (x-coordinates).
x_samples = samples[:, 0]
# `[:, 1]` selects all rows and the second column (y-coordinates).
y_samples = samples[:, 1]

# --- 2. Numerically Check for Dependence ---

# We will calculate the covariance between the x and y samples.
# For a symmetric distribution like a disk centered at the origin, we expect
# the covariance to be zero, meaning the variables are uncorrelated.

# First, calculate the sample mean for x and y.
# Due to symmetry, these should both be very close to zero.
mean_x = torch.mean(x_samples)
mean_y = torch.mean(y_samples)

# The formula for covariance is Cov(X, Y) = E[(X - E[X]) * (Y - E[Y])].
# We can approximate the expectation E[] by taking the mean over our samples.
covariance = torch.mean((x_samples - mean_x) * (y_samples - mean_y))

print(f"Distribution: {type(disk_dist).__name__} with radius {circle_radius}")
print(f"Number of samples: {num_samples}")
print(f"Numerical Covariance between X and Y: {covariance.item():.6f}")
print("(Theoretically, this should be 0 for a centered disk)")

"""
For a circular distribution centered at the origin, x and y are uncorrelated
(their covariance is zero), but they are NOT independent.
The value of x constrains the possible values of y (e.g., if x=r, then y must be 0).
We calculate the covariance here to numerically verify the non-correlation.
"""

# --- 3. Visualize the Results ---
plt.figure(figsize=(8, 8))
# Create a scatter plot of the generated (x, y) points.
# - `alpha=0.4` makes the points semi-transparent to show areas of high density.
# - `s=10` sets the size of the points.
plt.scatter(x_samples.numpy(), y_samples.numpy(), alpha=0.4, s=10)
plt.title(f"Sampling from a {type(disk_dist).__name__} Distribution")
plt.xlabel("X")
plt.ylabel("Y")
# `plt.axis('equal')` is crucial here. It ensures that the scaling on the x and y axes
# is the same, so that the circular shape is not distorted into an ellipse.
plt.axis('equal')
plt.grid(True)
# Display the final plot.
plt.show()