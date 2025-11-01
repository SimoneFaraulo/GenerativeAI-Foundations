import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from autoencoder import VAE


# --- 1. VAE Loss Function with Beta Parameter ---
def beta_vae_loss_function(recon_x, x, mu, log_var, beta):
    """
    Calculates the beta-VAE loss.
    A `beta` value of 1.0 represents the standard VAE loss, balancing
    reconstruction and regularization.
    """
    # Reconstruction Loss (MSE)
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # KL Divergence Loss
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Total Loss with beta weighting
    return recon_loss + beta * kl_loss


# --- 2. Data Generation (Semi-circle Manifold) ---
num_data_points = 2000
theta = torch.linspace(0, torch.pi, num_data_points)
x_data = torch.cos(theta)
y_data = torch.sin(theta)
data = torch.stack([x_data, y_data], dim=1)
data += torch.randn_like(data) * 0.05

# --- 3. Training a Balanced VAE ---
input_dim = 2
# Use a 2D latent space to match the setup in the other scripts.
# Even with a 2D latent space for a 1D manifold, a balanced beta
# should still encourage a well-structured latent space.
latent_dim = 2
learning_rate = 0.001
num_epochs = 2000
# A beta value around 1.0 is standard. A smaller value like 0.1 can also work well
# for simple datasets, prioritizing reconstruction while still providing enough
# regularization for good generation.
beta_value = 0.1

# Instantiate the VAE model.
model = VAE(input_dim, latent_dim)
# Define the optimizer.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Training a balanced beta-VAE with beta = {beta_value}...")
# The main training loop.
for epoch in range(num_epochs):
    # Forward pass
    reconstructions, mu, log_var = model(data)
    # Calculate the total VAE loss using our beta-weighted function.
    loss = beta_vae_loss_function(reconstructions, data, mu, log_var, beta=beta_value)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print the loss periodically.
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item() / len(data):.4f}')

# --- 4. Visualization ---
model.eval()
with torch.no_grad():
    # Get the latent space parameters for the original data.
    mu, log_var = model.encode(data)

    # Reconstruct the original data for comparison.
    reconstructed_data, _, _ = model(data)

    # Generate new data by sampling from the PRIOR distribution p(z) = N(0, 1).
    z_prior_samples = torch.randn(500, latent_dim)
    generated_data_from_prior = model.decode(z_prior_samples).numpy()

    # Get the latent representation of the input data for plotting.
    z_latent_samples = model.reparameterize(mu, log_var)

# --- Understanding the Balanced VAE Visualization ---
# The plots below will show a well-behaved VAE:
# 1. The latent space distribution q(z|x) will be a good approximation of the prior p(z).
#    It won't be a perfect match (like in posterior collapse), but it will be close,
#    indicating that the KL regularization was effective.
# 2. The reconstructions will be good, accurately capturing the semi-circle shape.
# 3. The generated samples will also be good, forming a coherent semi-circle because
#    the decoder was trained on a well-structured latent space.

# Create plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
fig.suptitle(f'Balanced VAE (beta = {beta_value})', fontsize=16)

# Plot 1: Latent Space
# The histogram will be a good approximation of the prior.
# With a 2D latent space, we use a scatter plot. The points should form a
# round cloud resembling a 2D standard normal distribution.
scatter = ax1.scatter(z_latent_samples[:, 0].numpy(), z_latent_samples[:, 1].numpy(), c=theta, cmap='viridis', s=10, alpha=0.7)
fig.colorbar(scatter, ax=ax1, label='Original Angle (theta)')
ax1.set_title('1. Latent Space (Well-Regularized)')
ax1.set_xlabel('Latent Dimension 1 (z1)')
ax1.set_ylabel('Latent Dimension 2 (z2)')
ax1.axis('equal')
ax1.grid(True)
ax1.text(0.5, -0.15,
         "The latent space approximates the prior,\ncreating a structured space for generation.",
         ha='center', transform=ax1.transAxes, style='italic', color='green')

# Plot 2: Reconstruction
# The reconstructions will be accurate.
ax2.scatter(data[:, 0], data[:, 1], s=10, alpha=0.5, label='Original Data')
ax2.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], s=20, alpha=0.8, color='orange',
            label='Reconstructed Data')
ax2.set_title('2. Good Reconstruction')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.axis('equal')
ax2.legend()
ax2.grid(True)

# Plot 3: Generation
# The generated samples will form a coherent semi-circle.
ax3.scatter(data[:, 0], data[:, 1], s=10, alpha=0.2, label='Original Data (for reference)')
ax3.scatter(generated_data_from_prior[:, 0], generated_data_from_prior[:, 1], s=20, alpha=0.8, color='red',
            label='Generated Data from Prior')
ax3.set_title('3. Good Generation')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.axis('equal')
ax3.legend()
ax3.grid(True)

# Display the final figure with all subplots.
plt.show()