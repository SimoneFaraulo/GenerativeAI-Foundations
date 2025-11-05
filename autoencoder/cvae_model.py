import torch
import torch.nn as nn

class ConditionalVAE(nn.Module):
    """
    Implementa un Conditional Variational Autoencoder (CVAE).
    
    Un CVAE estende il VAE permettendo di condizionare sia l'encoder che
    il decoder a un'informazione aggiuntiva 'c' (es. un'etichetta di classe,
    un attributo, o un altro dato).
    
    L'obiettivo è modellare p(x | c) imparando due distribuzioni:
    1. Encoder (Inference Network): q(z | x, c)
    2. Decoder (Generative Network): p(x | z, c)
    
    Come suggerito, un modo semplice ed efficace per implementare questo
    condizionamento è concatenare il tensore della condizione 'c' all'input 'x'
    per l'encoder, e al tensore latente 'z' per il decoder.
    """
    def __init__(self, input_dim, condition_dim, latent_dim):
        super(ConditionalVAE, self).__init__()
        
        # Le dimensioni ora dipendono dall'input e dalla condizione
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.latent_dim = latent_dim

        # --- Encoder: q(z | x, c) ---
        # L'input all'encoder è la concatenazione di x e c
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
        )
        # I lacer finali mappano l'output dell'encoder ai parametri
        # della distribuzione latente.
        self.fc_mu = nn.Linear(8, latent_dim)
        self.fc_log_var = nn.Linear(8, latent_dim)

        # --- Decoder: p(x | z, c) ---
        # L'input al decoder è la concatenazione di z e c
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + condition_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim) # L'output è la ricostruzione di x
        )

    def encode(self, x, c):
        """
        Produce i parametri (mu, log_var) della distribuzione
        latente approssimata q(z | x, c).
        
        Input:
          x: (N, input_dim) - I dati di input
          c: (N, condition_dim) - L'informazione di condizionamento
        Output:
          mu, log_var: (N, latent_dim) - Parametri della Gaussiana
        """
        # Concatena x e c lungo l'ultima dimensione (dim=1)
        xc = torch.cat([x, c], dim=1)
        
        # Passa l'input concatenato attraverso l'encoder
        h = self.encoder(xc)
        
        # Ritorna i parametri della distribuzione latente
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterize(self, mu, log_var):
        """
        Applica il Reparameterization Trick. Identico al VAE standard.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, c):
        """
        Ricostruisce l'input x dalla variabile latente z e
        dall'informazione di condizionamento c.
        
        Input:
          z: (N, latent_dim) - Il campione latente
          c: (N, condition_dim) - L'informazione di condizionamento
        Output:
          (N, input_dim) - L'input x ricostruito
        """
        # Concatena z e c lungo l'ultima dimensione (dim=1)
        zc = torch.cat([z, c], dim=1)
        
        # Passa l'input concatenato attraverso il decoder
        return self.decoder(zc)

    def forward(self, x, c):
        """
        Il forward pass completo del CVAE.
        """
        # 1. Encode: Ottieni i parametri da (x, c)
        mu, log_var = self.encode(x, c)
        
        # 2. Reparameterize: Ottieni z campionando da q(z | x, c)
        z = self.reparameterize(mu, log_var)
        
        # 3. Decode: Ricostruisci x da (z, c)
        reconstruction = self.decode(z, c)
        
        return reconstruction, mu, log_var