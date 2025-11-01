import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from autoencoder import VAE


# --- 2. VAE Loss Function ---
def vae_loss_function(recon_x, x, mu, log_var):
    """
    Calculates the VAE loss function, which is the negative of the Evidence Lower Bound (ELBO).
    The goal of training is to MAXIMIZE the ELBO, which is equivalent to MINIMIZING the negative ELBO.
    The loss is composed of two main terms:
    -ELBO = Reconstruction_Loss + KL_Divergence_Loss
    """
    # 1. Reconstruction Loss: -E_q(z|x)[log p(x|z)]
    # This term measures how well the decoder reconstructs the input data `x` from the latent code `z`.
    # For a Gaussian decoder where p(x|z) = N(x | g(z), beta*I), maximizing the log-likelihood
    # is equivalent to minimizing the Mean Squared Error (MSE) between the original and reconstructed data.
    # `reduction='sum'` sums the squared errors over all dimensions and all items in the batch.
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')

    # 2. KL Divergence Loss: KL( q(z|x) || p(z) )
    # This term acts as a regularizer. It forces the learned latent distribution `q(z|x)` (the encoder's output)
    # to be close to the prior distribution `p(z)`.
    #
    # Why is the prior p(z) a standard normal N(0, 1)?
    # We *choose* this simple distribution for two main reasons:
    #   a) It's easy to sample from (e.g., `torch.randn()`), which is essential for generating new data.
    #   b) The KL divergence between two Gaussians has a simple, analytical formula, which makes the loss
    #      easy to compute and differentiate.
    # We could have chosen a different prior, like N(10, 0.5), but it would complicate the math
    # without adding any benefit. The goal is simply to enforce a consistent, organized structure.
    # This regularization encourages a well-structured, continuous latent space that is suitable for generation.
    # For two Gaussian distributions, this KL divergence has a closed-form analytical solution:
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # The total loss is the sum of the reconstruction and KL divergence terms.
    # A `beta` hyperparameter can be introduced here to balance the two terms (as in a beta-VAE),
    # but here we use the standard VAE formulation where beta is implicitly 1.
    return recon_loss + kl_loss


# --- 3. Data Generation (Semi-circle Manifold) ---
# We create a simple 1D manifold (a semi-circle) embedded in 2D space.
num_data_points = 2000
# Generate angles from 0 to pi.
theta = torch.linspace(0, torch.pi, num_data_points)
# Convert angles to (x, y) coordinates.
x_data = torch.cos(theta)
y_data = torch.sin(theta)
# Stack the coordinates to create our dataset of shape (2000, 2).
data = torch.stack([x_data, y_data], dim=1)
# Add a small amount of Gaussian noise to make the data more realistic.
data += torch.randn_like(data) * 0.05

# --- 4. Training the VAE ---
# Define the dimensions for our network.
input_dim = 2
latent_dim = 1 # The manifold is 1D, so we use a 1D latent space.
# Define the hyperparameters for the training process.
learning_rate = 0.001
num_epochs = 2000

# Instantiate the VAE model.
model = VAE(input_dim, latent_dim)
# Define the optimizer. Adam is a popular and effective choice.
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print("Training a Variational Autoencoder (VAE)...")
# The main training loop.
for epoch in range(num_epochs):
    # --- Forward pass ---
    # Pass the data through the model to get the reconstruction and the latent parameters.
    reconstructions, mu, log_var = model(data)
    # Calculate the total VAE loss using our defined function.
    loss = vae_loss_function(reconstructions, data, mu, log_var)

    # --- Backward pass and optimization ---
    # Zero out the gradients from the previous iteration.
    optimizer.zero_grad()
    # Perform backpropagation to compute the gradients of the loss.
    loss.backward()
    # Update the model's weights using the computed gradients.
    optimizer.step()

    # Print the loss periodically to monitor training progress.
    # We divide by the number of data points to get the average loss per sample.
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item() / len(data):.4f}')

# --- 5. Visualization ---
# Set the model to evaluation mode. This disables layers like Dropout if they were present.
model.eval()
# Use `torch.no_grad()` to disable gradient calculation, which speeds up inference and saves memory.
with torch.no_grad():
    # Get the latent space parameters (mu and log_var) for the original data.
    mu, log_var = model.encode(data)

    # --- Reconstruct the original data for comparison ---
    # This shows how well the VAE can encode and then decode the input data.
    reconstructed_data, _, _ = model(data)

    # --- Generate new data by sampling from the PRIOR distribution p(z) = N(0, 1) ---
    # This is the key to generation: we sample random points from the simple, known prior...
    z_prior_samples = torch.randn(500, latent_dim)  # Sample 500 points from N(0, 1)
    # ...and then pass them through the decoder to generate new data points.
    generated_data_from_prior = model.decode(z_prior_samples).numpy()

    # --- Get the latent representation of the input data for plotting ---
    # We use the reparameterization trick to get a sample `z` for each input `x`.
    # This shows us where the input data is mapped in the latent space.
    z_latent_samples = model.reparameterize(mu, log_var)

# --- Understanding the Latent Space Visualization ---
# A common point of confusion is thinking that the encoder learns to output mu=0 and log_var=0
# for every input. This is NOT the case.
#
# The VAE loss creates a "tug-of-war":
# 1. The Reconstruction Loss pushes the encoder to find specific `mu` and `log_var` values
#    that best describe each individual input `x`. For a specific `x_i`, the distribution
#    q(z|x_i) will be N(mu_i, var_i), which is NOT N(0, 1).
# 2. The KL Divergence Loss pushes the encoder to make the distribution `q(z|x)` (defined
#    by `mu` and `log_var`) as close as possible to the prior `p(z) = N(0, 1)`.
#
# The final result is that the encoder learns to output *different* `mu` and `log_var` for each `x`,
# but the *aggregate distribution* of all the sampled `z` values across the entire dataset
# will resemble the standard normal prior. The histogram below visualizes this aggregate distribution,
# confirming that the KL regularization has successfully organized the latent space.

# Create plots
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
fig.suptitle('Variational Autoencoder (VAE): Latent Space and Generation', fontsize=16)

# Plot 1: Latent Space
# Plot a histogram of the latent representations of the input data.
# This shows the distribution of `q(z|x)`.
ax1.hist(z_latent_samples.numpy(), bins=50, density=True, label='Learned Latent Distribution q(z|x)')
# Plot the target prior distribution p(z) = N(0, 1) for comparison
x_axis = torch.linspace(-3, 3, 200)
prior_pdf = torch.exp(torch.distributions.Normal(0, 1).log_prob(x_axis))
# If training was successful, the histogram should closely match the red dashed line.
ax1.plot(x_axis, prior_pdf, 'r--', label='Target Prior p(z) = N(0,1)')
ax1.set_title('1. Latent Space of VAE')
ax1.set_xlabel('Latent Dimension (z)')
ax1.set_ylabel('Density')
ax1.legend()
ax1.grid(True)
# Add an explanatory text box.
ax1.text(0.5, -0.15,
         "The latent space is regularized to match the simple prior.\nThis makes it possible to sample from.",
         ha='center', transform=ax1.transAxes, style='italic')

# Plot 2: Reconstruction
# This plot shows the original data and how well the VAE reconstructs it.
ax2.scatter(data[:, 0], data[:, 1], s=10, alpha=0.5, label='Original Data')
ax2.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], s=20, alpha=0.8, color='orange',
            label='Reconstructed Data')
ax2.set_title('2. Reconstruction of Original Data')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
# Ensure the axes have the same scale to avoid distorting the semi-circle.
ax2.axis('equal')
ax2.legend()
ax2.grid(True)

# Plot 3: Generation
# This plot shows the newly generated data, created from random samples of the prior.
ax3.scatter(data[:, 0], data[:, 1], s=10, alpha=0.2, label='Original Data (for reference)')
ax3.scatter(generated_data_from_prior[:, 0], generated_data_from_prior[:, 1], s=20, alpha=0.8, color='red',
            label='Generated Data from Prior')
ax3.set_title('3. Generation from Prior Samples')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.axis('equal')
ax3.legend()
ax3.grid(True)

# Display the final figure with both subplots.
plt.show()