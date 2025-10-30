from torch import nn

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