import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt # Note: `import math` is not needed as torch.pi is used.
from autoencoder import Autoencoder

# --- 1. Generate Data on a 2D Manifold in 3D Space ---
# The idea is to create 3D points (x, y, z) that lie on a 2D surface (a manifold).
# We will use the "sombrero" function z = sin(sqrt(x^2 + y^2)) for this.
num_data_points = 2000

# To create the (x, y) domain for our function, we will sample points uniformly from a 2D disk.
# This ensures a good, dense distribution of points on the resulting surface.
radius = 5.0
# Sample the angle `theta` uniformly from [0, 2*pi].
theta = torch.rand(num_data_points) * 2 * torch.pi
# Sample the radius `r` correctly to achieve a uniform spatial density.
# This is done by taking the square root of a uniform random variable, which compensates
# for the fact that there is more area at larger radii.
r = radius * torch.sqrt(torch.rand(num_data_points))
# Convert from polar (r, theta) to Cartesian (x, y) coordinates.
x_data = r * torch.cos(theta)
y_data = r * torch.sin(theta)

# Calculate the z coordinate using our function f(x, y).
# This maps the 2D disk domain onto a 3D surface.
z_data = torch.sin(torch.sqrt(x_data**2 + y_data**2))

# Stack the coordinates to get a tensor of shape (num_data_points, 3).
# This is our 3D dataset that lies on a 2D manifold.
data = torch.stack([x_data, y_data, z_data], dim=1)


# --- 2. Configure the Autoencoder and Training ---
# Define the dimensions for our network.
input_dimension = 3  # Our data is (x, y, z)
latent_dimension = 2 # We want to compress the 2D surface (manifold) to 2 dimensions.
# Define the hyperparameters for the training process.
learning_rate = 0.001
num_epochs = 2000

# Instantiate the model.
model = Autoencoder(input_dimension, latent_dimension)
# Define the loss function. Mean Squared Error (MSE) is the standard choice for autoencoders,
# as it measures the average squared difference between the input and its reconstruction.
criterion = nn.MSELoss()
# Define the optimizer. Adam is a popular and effective general-purpose optimizer for neural networks.
# It will adjust the model's weights to minimize the loss.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 3. Training Loop ---
print("Starting Autoencoder training...")
for epoch in range(num_epochs):
    # --- Forward pass ---
    # Pass the entire dataset through the model to get the reconstructions.
    reconstructions = model(data)
    # Calculate the loss by comparing the reconstructions to the original data.
    loss = criterion(reconstructions, data)

    # --- Backward pass and optimization ---
    optimizer.zero_grad() # Zero out the gradients from the previous iteration.
    loss.backward()       # Perform backpropagation to compute the gradients of the loss with respect to the model's weights.
    optimizer.step()      # Update the model's weights using the computed gradients.

    # Print the loss periodically to monitor the training progress.
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')

print("Training complete.")

# --- 4. Visualize the Results ---
# Set the model to evaluation mode. This disables layers like Dropout or BatchNorm if they were present.
model.eval()
# Use `torch.no_grad()` to disable gradient calculation, which speeds up inference and saves memory.
with torch.no_grad():
    # Get both the reconstructed data and the latent space representations in a single forward pass.
    reconstructed_data = model(data).numpy()
    latent_representations = model.encoder(data).numpy()

# Convert the original data tensor to a NumPy array for plotting.
original_data = data.numpy()

# Create a figure that will contain both plots
fig = plt.figure(figsize=(24, 8))
fig.suptitle('Autoencoder Analysis', fontsize=16)

# --- Subplot 1: Original 3D Data ---
ax1 = fig.add_subplot(1, 3, 1, projection='3d')
# Create a 3D scatter plot of the original data points.
ax1.scatter(original_data[:, 0], original_data[:, 1], original_data[:, 2], s=20, alpha=0.5, label='Original Data', color='blue')
# Set the title and labels for the 3D plot.
ax1.set_title('Original 3D Data')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()

# --- Subplot 2: Reconstructed 3D Data ---
ax2 = fig.add_subplot(1, 3, 2, projection='3d')
ax2.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], reconstructed_data[:, 2], s=20, alpha=0.5, label='Reconstructed Data', color='red')
ax2.set_title('Reconstructed 3D Data')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend()

# --- Subplot 3: 2D Latent Space Plot ---
# Add the third subplot to the figure for the 2D latent space.
ax3 = fig.add_subplot(1, 3, 3)
# Create a 2D scatter plot of the latent representations (z1, z2).
# `c=r.numpy()`: This is a key part of the visualization. We color each point in the
# latent space based on its original radius `r` in the (x, y) plane. This helps us
# see if the encoder has learned to organize the data in a meaningful way.
# `cmap='viridis'`: A color map to visualize the radius values.
scatter = ax3.scatter(latent_representations[:, 0], latent_representations[:, 1], c=r.numpy(), cmap='viridis', s=20, alpha=0.7)
# Add a color bar to the plot to show the mapping between colors and radius values.
fig.colorbar(scatter, ax=ax3, label='Original Radius (r)')
# Set the title and labels for the 2D plot.
ax3.set_title('Latent Space Representation (2D)')
ax3.set_xlabel('Latent Dimension 1 (z1)')
ax3.set_ylabel('Latent Dimension 2 (z2)')
# Ensure the axes have the same scale to avoid distortion.
ax3.axis('equal')
ax3.grid(True)

# Display the entire figure with both subplots.
plt.show()
