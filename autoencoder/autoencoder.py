from torch import nn
import torch

# --- Define the Autoencoder Model ---
# An autoencoder is composed of an encoder and a decoder.
# The encoder maps the input (3D) to a latent space (2D).
# The decoder maps the latent space (2D) back to the output space (3D), trying to reconstruct the input.
class Autoencoder(nn.Module):
    # This class defines our autoencoder architecture, inheriting from PyTorch's base Module.
    def __init__(self, input_dim, hidden_dim=128, latent_dim=32):
        # The constructor initializes the layers of the network.
        super(Autoencoder, self).__init__()
        # The encoder part of the network. It's a sequence of layers.
        self.encoder = nn.Sequential(
            # A linear layer that maps the input dimension (3) to a larger hidden dimension (128).
            nn.Linear(input_dim, hidden_dim),
            # A ReLU activation function to introduce non-linearity, allowing the network to learn complex mappings.
            nn.ReLU(),
            # A final linear layer that maps the hidden dimension (128) to the compressed latent dimension (2).
            nn.Linear(hidden_dim, latent_dim)
        )
        # The decoder part of the network. It has a symmetric structure to the encoder.
        self.decoder = nn.Sequential(
            # A linear layer that maps the latent dimension (2) back to a larger hidden dimension (128).
            nn.Linear(latent_dim, hidden_dim),
            # A ReLU activation function.
            nn.ReLU(),
            # A final linear layer that maps the hidden dimension (128) back to the original input dimension (3).
            # No activation function is used here to allow the output to have any real value.
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        # The forward method defines how data flows through the network.
        # First, the input `x` is passed through the encoder to get the latent representation.
        latent = self.encoder(x)
        # Then, the latent representation is passed through the decoder to get the reconstructed output.
        reconstruction = self.decoder(latent)
        return reconstruction

# --- The Variational Autoencoder (VAE) and Amortized Inference ---
# The VAE introduces a probabilistic spin on the standard autoencoder, turning it
# into a powerful generative model. A key concept that makes VAEs practical is
# **Amortized Inference**, which is implemented by the encoder.
#
# The Problem (The "Impractical" Way):
# For each data point `x_n` in our dataset, there is an ideal but intractable posterior
# distribution `p(z | x_n)` that describes the perfect latent representation. A VAE
# tries to approximate this with a simpler, tractable distribution, `q_n(z)`.
# A naive approach would be to optimize a separate `q_n(z)` (with its own set of
# parameters, like `mu_n` and `sigma_n`) for *every single data point*. For a dataset
# with millions of points, this would mean optimizing millions of separate parameter
# sets, which is computationally infeasible. It's like building a tiny, custom model
# for each individual data point.
#
# The Solution (Amortized Inference):
# Instead of learning millions of individual models, we train a single, powerful neural
# network—the **encoder**—to act as a universal function. This encoder, parameterized
# by a single set of weights `phi`, takes any data point `x` as input and instantly
# outputs the parameters for its corresponding latent distribution `q(z | x, phi)`.
# The cost of inference is thus "amortized" across all data points, as the single
# encoder learns a general mapping from the data space to the latent parameter space.
# This is why the VAE encoder doesn't output `z` directly, but rather `mu` and `log_var`.

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        # --- Encoder ---
        # The encoder maps the input x to the parameters of the latent distribution q(z|x)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # This layer outputs the mean (mu) of the latent Gaussian
        self.fc_mu = nn.Linear(32, latent_dim)
        # This layer outputs the log-variance (log_var) of the latent Gaussian
        self.fc_log_var = nn.Linear(32, latent_dim)

        # --- Note on the Probabilistic Decoder ---
        # In pure theory, the decoder should also be probabilistic. It should output
        # the parameters of a distribution p(x|z), typically a Gaussian N(g(z), beta*I).
        # The output of the neural network `g(z)` would be the mean of this Gaussian.
        #
        # However, in most common VAE implementations (including this one), a simplification
        # is made: we assume the variance `beta` is a fixed, small constant.
        # When `beta` is fixed, maximizing the log-likelihood log p(x|z) becomes
        # equivalent to minimizing the Mean Squared Error (MSE) between the input `x`
        # and the decoder's output `g(z)`.
        #
        # Therefore, our decoder is trained to directly output the mean of the reconstruction,
        # and we use MSE as our reconstruction loss. This is why the decoder below
        # is deterministic and only outputs a single tensor.

        # --- Decoder ---
        # The decoder maps a point z from the latent space back to the data space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def encode(self, x):
        # Pass input through the shared encoder layers
        h = self.encoder(x)
        # Return the mean and log-variance
        return self.fc_mu(h), self.fc_log_var(h)

    # --- The Reparameterization Trick ---
    # A crucial problem arises when training a VAE. The loss function depends on `z`,
    # which is *sampled* from the distribution `q(z | x, phi)` defined by the encoder's outputs (`mu`, `log_var`).
    #
    # The Problem (The "Stochastic Node"):
    # The sampling operation `z = sample_from_gaussian(mu, log_var)` is a random, or "stochastic,"
    # process. Backpropagation requires a deterministic path of differentiable operations to calculate
    # gradients. A random sampling node breaks this path. You cannot directly calculate how a small
    # change in `mu` or `log_var` would change the output of a *random* sample. The gradient
    # cannot flow "through" the sampling operation back to the encoder.
    #
    # The Solution (Moving the Randomness):
    # The reparameterization trick solves this by reframing the sampling process. Instead of sampling
    # `z` directly from a distribution dependent on `mu` and `log_var`, we do the following:
    # 1. Sample a random variable `epsilon` from a fixed, simple distribution, typically the
    #    standard normal `N(0, 1)`. This random input is external and does not depend on any
    #    network parameters.
    # 2. Calculate `z` using a deterministic function: `z = mu + std * epsilon`, where `std` is
    #    derived from `log_var`.
    #
    # Why it works:
    # The path from the encoder's outputs (`mu`, `log_var`) to `z` now consists only of
    # simple, differentiable arithmetic operations (multiplication and addition). The randomness
    # has been "injected" from the outside. This creates a differentiable path from the final
    # loss all the way back to the encoder's weights, allowing backpropagation to work correctly.

    def reparameterize(self, mu, log_var):
        # This is the reparameterization trick.
        # It allows us to sample from N(mu, var) in a way that is differentiable.
        std = torch.exp(0.5 * log_var)  # Calculate standard deviation from log-variance
        eps = torch.randn_like(std)  # Sample epsilon from a standard normal N(0, 1)
        return mu + eps * std  # Scale and shift epsilon to get z

    def decode(self, z):
        # Pass the latent code z through the decoder to get the reconstruction
        return self.decoder(z)

    def forward(self, x):
        # The full forward pass of the VAE
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var
