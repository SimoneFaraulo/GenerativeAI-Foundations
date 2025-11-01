import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from autoencoder import VAE


# --- 1. VAE Loss Function with Beta Parameter ---
def beta_vae_loss_function(recon_x, x, mu, log_var, beta):
    """
    Calculates the beta-VAE loss.
    A `beta` value greater than 1.0 heavily penalizes the KL divergence,
    which can intentionally induce posterior collapse for demonstration purposes.
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

# --- 3. Training the VAE with High Beta to Induce Collapse ---
input_dim = 2
latent_dim = 1
learning_rate = 0.001
num_epochs = 2000
# CRUCIAL: Set beta to a high value to force posterior collapse.
# This tells the model that satisfying the KL term is much more important
# than the reconstruction quality.
beta_value = 10.0

# Instantiate the VAE model.
model = VAE(input_dim, latent_dim)
# Define the optimizer.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Training a beta-VAE with beta = {beta_value} to demonstrate posterior collapse...")
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

# --- Understanding the Posterior Collapse Visualization ---
# The plots below will show the classic signs of posterior collapse:
# 1. The latent space distribution q(z|x) will perfectly match the prior p(z).
#    This looks good, but it's bad because it means the encoder has learned to
#    ignore the input `x`. The latent code contains no useful information.
# 2. The reconstructions will be poor and blurry. Since the decoder receives
#    almost the same latent code `z` (sampled from the prior) for every input `x`,
#    it learns to output the "safest" option, which is the average of the entire dataset.

# Create plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
fig.suptitle(f'VAE Posterior Collapse (beta = {beta_value})', fontsize=16)

# Plot 1: Latent Space
# The histogram will perfectly match the prior, showing the collapse.
ax1.hist(z_latent_samples.numpy(), bins=50, density=True, label='Learned Latent Distribution q(z|x)')
x_axis = torch.linspace(-3, 3, 200)
prior_pdf = torch.exp(torch.distributions.Normal(0, 1).log_prob(x_axis))
ax1.plot(x_axis, prior_pdf, 'r--', label='Target Prior p(z) = N(0,1)')
ax1.set_title('1. Latent Space (Collapsed)')
ax1.set_xlabel('Latent Dimension (z)')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True)
ax1.text(0.5, -0.15,
         "The encoder ignores the input;\nq(z|x) has collapsed to the prior p(z).",
         ha='center', transform=ax1.transAxes, style='italic', color='red')

# Plot 2: Reconstruction
# The reconstructions will be a blurry mess, clustered around the data mean.
ax2.scatter(data[:, 0], data[:, 1], s=10, alpha=0.5, label='Original Data')
ax2.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], s=20, alpha=0.8, color='orange',
            label='Reconstructed Data')
ax2.set_title('2. Poor Reconstruction')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.axis('equal')
ax2.legend()
ax2.grid(True)

# Plot 3: Generation
# The generated samples will also be a blurry mess, similar to the reconstructions.
ax3.scatter(data[:, 0], data[:, 1], s=10, alpha=0.2, label='Original Data (for reference)')
ax3.scatter(generated_data_from_prior[:, 0], generated_data_from_prior[:, 1], s=20, alpha=0.8, color='red',
            label='Generated Data from Prior')
ax3.set_title('3. Poor Generation')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.axis('equal')
ax3.legend()
ax3.grid(True)

# Display the final figure with all subplots.
plt.show()