import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from autoencoder import Autoencoder

# --- 1. Generate Data on a 2D Manifold in 3D Space ---
# The idea is to create 3D points (x, y, z) that lie on a 2D surface (a manifold).
# We will generate data on a simple flat plane defined by z = a*x + b*y.
num_data_points = 20000

# To create the (x, y) domain for our function, we will sample points uniformly from a 2D disk.
# This ensures a good, dense distribution of points on the resulting surface.
radius = 5.0
# Sample the angle `theta` uniformly from [0, 2*pi].
theta = torch.rand(num_data_points) * 2 * torch.pi
# Sample the radius `r` correctly to achieve a uniform spatial density.
r = radius * torch.sqrt(torch.rand(num_data_points))
# Convert from polar (r, theta) to Cartesian (x, y) coordinates.
x_data = r * torch.cos(theta)
y_data = r * torch.sin(theta)

# Calculate the z coordinate using a simple linear plane equation.
# This creates a 2D flat manifold embedded in 3D space.
z_data = 0.5 * x_data + 0.2 * y_data

# Stack the coordinates to get a tensor of shape (num_data_points, 3).
# This is our original, clean 3D dataset that lies on a 2D manifold.
data = torch.stack([x_data, y_data, z_data], dim=1)

# --- 2. Add Gaussian Noise to the Data ---
# This is the key step for a denoising autoencoder. We corrupt the clean data.
# The autoencoder will be trained to reconstruct the original `data` from `noisy_data`.
noise_std = 2.0  # Standard deviation of the Gaussian noise. Controls the noise intensity.

# Generate noise with the same shape as the data and add it.
# `torch.randn_like(data)` creates random numbers from a standard normal distribution.
noise = torch.randn_like(data) * noise_std
noisy_data = data + noise

# --- 3. Configure the Autoencoder and Training ---
# Define the dimensions for our network.
input_dimension = 3  # Our data is (x, y, z)
latent_dimension = 2  # We want to compress the 2D surface (manifold) to 2 dimensions.
# Define the hyperparameters for the training process.
learning_rate = 0.005
num_epochs = 2000

# Instantiate the model.
# The `Autoencoder` class is assumed to be defined in `autoencoder.py`.
model = Autoencoder(input_dimension, hidden_dim=9, latent_dim=latent_dimension)
# Define the loss function. Mean Squared Error (MSE) is the standard choice for autoencoders.
criterion = nn.MSELoss()
# Define the optimizer. Adam is a popular and effective choice.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- 4. Training Loop ---
print("Starting Denoising Autoencoder training...")
for epoch in range(num_epochs):
    # Pass the NOISY data through the model to get the reconstructions.
    reconstructions = model(noisy_data)
    # Calculate the loss by comparing the `reconstructions` to the ORIGINAL CLEAN `data`.
    # This is the core principle of a denoising autoencoder.
    loss = criterion(reconstructions, data)

    # Backward pass and optimization
    optimizer.zero_grad()  # Zero out the gradients from the previous iteration.
    loss.backward()  # Perform backpropagation to compute gradients.
    optimizer.step()  # Update the model's weights.

    # Print the loss periodically to monitor the training progress.
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("Training complete.")

# --- 5. Visualize the Results ---
# Set the model to evaluation mode.
model.eval()
# Use `torch.no_grad()` to disable gradient calculation for efficiency.
with torch.no_grad():
    # Get the reconstructed data by feeding the NOISY data to the model.
    reconstructed_data = model(noisy_data).numpy()
    # Get the latent space representations by passing the NOISY data through the encoder.
    latent_representations = model.encoder(noisy_data).numpy()
    
    # --- Genera dati campionando dallo spazio latente ---
    # Stiamo campionando da una Gaussiana

    # Determina la scala del campionamento in base alle rappresentazioni latenti reali
    latent_mean = torch.mean(torch.from_numpy(latent_representations), axis=0)
    latent_std = torch.std(torch.from_numpy(latent_representations), axis=0)
    
    # Campiona da una distribuzione simile a quella dei dati latenti
    z_gen = (torch.randn(500, latent_dimension) * latent_std) + latent_mean 
    generated_data = model.decoder(z_gen).numpy()

# Convert tensors to NumPy arrays for plotting.
original_data_np = data.numpy()
noisy_data_np = noisy_data.numpy()

# Create a figure that will contain all plots.
fig = plt.figure(figsize=(40, 8))
fig.suptitle('Denoising Autoencoder Analysis', fontsize=16)

# --- Subplot 1: Original Clean 3D Data ---
ax1 = fig.add_subplot(1, 5, 1, projection='3d')
ax1.scatter(original_data_np[:, 0], original_data_np[:, 1], original_data_np[:, 2], s=15, alpha=0.5, label='Original Clean Data', color='blue')
ax1.set_title('1. Original Clean Data')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
ax1.legend()

# --- Subplot 2: Noisy Input 3D Data ---
ax2 = fig.add_subplot(1, 5, 2, projection='3d')
ax2.scatter(noisy_data_np[:, 0], noisy_data_np[:, 1], noisy_data_np[:, 2], s=15, alpha=0.5, label='Noisy Input Data', color='gray')
ax2.set_title('2. Noisy Input Data')
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
ax2.legend()

# --- Subplot 3: Reconstructed (Denoised) 3D Data ---
ax3 = fig.add_subplot(1, 5, 3, projection='3d')
ax3.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], reconstructed_data[:, 2], s=15, alpha=0.7, label='Reconstructed (Denoised) Data', color='red')
ax3.set_title('3. Reconstructed (Denoised) Data')
ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
ax3.legend()

# --- Subplot 4: 2D Latent Space Plot ---
ax4 = fig.add_subplot(1, 5, 4)
scatter = ax4.scatter(latent_representations[:, 0], latent_representations[:, 1], c=r.numpy(), cmap='viridis', s=20, alpha=0.7)
fig.colorbar(scatter, ax=ax4, label='Original Radius (r)')
ax4.set_title('4. Latent Space of Noisy Data')
ax4.set_xlabel('Latent Dimension 1 (z1)')
ax4.set_ylabel('Latent Dimension 2 (z2)')
ax4.axis('equal')
ax4.grid(True)

# --- 4: Generated Data ---
ax4 = fig.add_subplot(1, 5, 5, projection='3d')
ax4.scatter(generated_data[:, 0], generated_data[:, 1], generated_data[:, 2], s=20, alpha=0.5, label='Generated Data', color='green')
ax4.set_title('4. Generated Data (from random latent samples)')
ax4.set_xlabel('X') 
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')


# Display the entire figure with all subplots.
plt.show()