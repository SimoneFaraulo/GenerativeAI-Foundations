import torch
import matplotlib.pyplot as plt

# --- 1. Define the Square's Boundaries and Parameters ---
# We'll define a square centered at the origin.
side_length = 2.0
# `num_samples` defines how many random points we initially generate within the square.
num_samples = 10000 # Increased for better visual density after filtering.

# Define the parameters for the circle that we will use to filter the square samples.
# `circle_r` is the radius of this target circle.
circle_r = 1.0
# `center_x`, `center_y` define the coordinates of the circle's center.
center_x, center_y = 0.0, 0.0

# The square will extend from -1 to 1 on both the x and y axes.
low_bound = -side_length / 2
high_bound = side_length / 2

# --- 2. Create the 2D Uniform Distribution for the Square ---
# For a square aligned with the axes, the x and y coordinates are independent.
# We can create a single `Uniform` distribution object that handles both dimensions
# by providing the low and high bounds for each dimension as tensors.

# Low bounds for [x, y]
low_tensor = torch.tensor([low_bound, low_bound])
# High bounds for [x, y]
high_tensor = torch.tensor([high_bound, high_bound])

# Create the 2D uniform distribution object. This is equivalent to two separate
# Uniform distributions, one for x and one for y.
square_dist = torch.distributions.Uniform(low=low_tensor, high=high_tensor)

# --- 3. Generate Samples ---
# Now we can simply call the .sample() method to get our 2D points.
# The output will be a tensor of shape (num_samples, 2).
samples = square_dist.sample(sample_shape=(num_samples,))

# --- 4. Filter Samples to be within a Circle (Rejection Sampling Logic) ---
# This section implements the core logic of filtering. We generate samples from a simple
# proposal distribution (the square) and then accept only those that fall within our
# target distribution (the circle).

# Calculate the squared Euclidean distance of each generated sample from the circle's center.
# This is done in a vectorized manner for efficiency, avoiding explicit Python loops.
# `samples[:, 0]` gets all x-coordinates, `samples[:, 1]` gets all y-coordinates.
distances_sq = (samples[:, 0] - center_x)**2 + (samples[:, 1] - center_y)**2

# Create a boolean mask. This mask is `True` for any sample whose squared distance
# from the center is less than or equal to the squared radius of the circle.
# This effectively identifies all points that are inside or on the boundary of the circle.
inside_circle_mask = distances_sq <= circle_r**2

# Use the boolean mask to select only the samples that fall within the circle.
# `circle_samples` will now contain only the points that are inside the defined circle.
circle_samples = samples[inside_circle_mask]

# --- 5. Visualize the Results ---
plt.figure(figsize=(8, 8))
# First, plot all samples initially generated within the square as a background.
plt.scatter(samples[:, 0].numpy(), samples[:, 1].numpy(), alpha=0.1, s=10, color='blue', label='Samples in Square (Rejected)')

# Then, plot the filtered samples that are inside the circle on top.
plt.scatter(circle_samples[:, 0].numpy(), circle_samples[:, 1].numpy(), alpha=0.6, s=10, color='red', label=f'Samples in Circle (Accepted)')

plt.title(f"Filtering Square Samples for a Circle (Rejection Sampling)")
plt.xlabel("X")
plt.ylabel("Y")
# `plt.axis('equal')` ensures the square is not distorted by the plot's aspect ratio.
plt.axis('equal')
# Display the labels defined in the scatter plots.
plt.legend()
plt.grid(True)
plt.show()
