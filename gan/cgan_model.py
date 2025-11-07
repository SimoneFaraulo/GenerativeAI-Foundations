import torch
import torch.nn as nn

# --- 1. Definizione del Generatore Condizionale (CGAN) ---
class Generator(nn.Module):
    """
    Implementa il Generatore di una CGAN.
    
    A differenza di un GAN standard, il Generatore riceve due input:
    1. Un vettore di rumore latente 'z'.
    2. Un'informazione di condizionamento 'c' (es. un'etichetta di classe).
    
    L'obiettivo è imparare a generare un output x che sia plausibile
    E che corrisponda alla condizione 'c'.
    
    p(x | z, c)
    
    Il modo più semplice per implementarlo è concatenare 'z' e 'c' e
    usarli come input per una rete neurale standard.
    """
    def __init__(self, latent_dim, condition_dim, output_dim):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            # L'input è la concatenazione di z e c
            nn.Linear(latent_dim + condition_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            # L'output ha la dimensione dei dati (es. 2 per dati 2D)
            nn.Linear(128, output_dim)
        )

    def forward(self, z, c):
        # Concatena il vettore latente 'z' e la condizione 'c' lungo la dimensione 1
        # Input: z (N, latent_dim), c (N, condition_dim)
        # Output: zc (N, latent_dim + condition_dim)
        zc = torch.cat([z, c], dim=1)
        
        # Passa l'input concatenato attraverso il modello
        return self.model(zc)


# --- 2. Definizione del Discriminatore Condizionale (CGAN) ---
class Discriminator(nn.Module):
    """
    Implementa il Discriminatore di una CGAN.
    
    A differenza di un GAN standard, il Discriminatore riceve due input:
    1. Un campione di dati 'x' (reale o sintetico).
    2. L'informazione di condizionamento 'c' corrispondente.
    
    L'obiettivo è imparare a rispondere a una domanda più complessa:
    "Dato che mi aspetto un campione della classe 'c', questo campione 'x'
    è reale E appartiene a quella classe?"
    
    p(y=real | x, c)
    
    Anche qui, l'implementazione più semplice è concatenare 'x' e 'c'.
    """
    def __init__(self, input_dim, condition_dim):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            # L'input è la concatenazione di x e c
            nn.Linear(input_dim + condition_dim, 128),
            nn.LeakyReLU(0.2, inplace=True), # LeakyReLU è comune nelle GAN
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2, inplace=True),
            # L'output è un singolo logit
            nn.Linear(128, 1),
            # Sigmoid per mappare l'output a una probabilità [0, 1]
            nn.Sigmoid()
        )

    def forward(self, x, c):
        # Concatena il campione 'x' e la condizione 'c'
        # Input: x (N, input_dim), c (N, condition_dim)
        # Output: xc (N, input_dim + condition_dim)
        xc = torch.cat([x, c], dim=1)
        
        # Passa l'input concatenato attraverso il modello
        return self.model(xc)