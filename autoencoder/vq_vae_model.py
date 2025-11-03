import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. Vector Quantizer (Codebook) ---
# Questo modulo è il cuore del VQ-VAE. Prende l'output continuo dell'encoder
# (z_e) e lo mappa al vettore più vicino nel codebook (z_q).
class VectorQuantizer(nn.Module):
    """
    Implementa il "codebook" o "spazio di embedding" discreto.
    Corrisponde all'operazione q(z|x) che mappa l'output continuo dell'encoder
    z_e(x) all'indice k del vettore e_k più vicino nel codebook E.
    """
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.K = num_embeddings    # K: Numero di vettori nel codebook (es. 512)
        self.D = embedding_dim     # D: Dimensione di ogni vettore (es. 64)
        
        # Inizializza il codebook (la matrice di embedding E)
        # E è un tensore di forma (K, D) che verrà appreso.
        self.embedding = nn.Embedding(self.K, self.D)
        # Inizializziamo i pesi in modo uniforme
        self.embedding.weight.data.uniform_(-1.0 / self.K, 1.0 / self.K)

    def forward(self, z_e):
        """
        Forward pass: mappa z_e a z_q.
        Input:
          z_e: (N, D, H, W) - Output continuo dell'encoder
        Output:
          z_q: (N, D, H, W) - Vettori quantizzati (copiati dal codebook)
          loss: Scalare - La loss per aggiornare il codebook (VQ loss)
          indices: (N, H, W) - Gli indici k scelti per ogni posizione
        """
        # 1. Preparazione di z_e
        # L'output dell'encoder (N, D, H, W) deve essere confrontato con E (K, D).
        # Riordiniamo z_e in (N*H*W, D)
        N, D, H, W = z_e.shape
        z_e_flat = z_e.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        
        # 2. Calcolo delle distanze
        # Dobbiamo trovare per ogni vettore in z_e_flat (dim D) il vettore
        # più vicino nel codebook E (K, D).
        # Usiamo un trucco per calcolare la distanza euclidea al quadrato:
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2(a @ b)
        
        # ||a||^2: Calcola la norma al quadrato per ogni vettore in z_e_flat
        z_e_norm_sq = torch.sum(z_e_flat**2, dim=1, keepdim=True) # (N*H*W, 1)
        # ||b||^2: Calcola la norma al quadrato per ogni vettore nel codebook
        e_norm_sq = torch.sum(self.embedding.weight**2, dim=1) # (K,)
        # 2(a @ b): Prodotto scalare tra z_e_flat e i vettori del codebook
        e_dot_z = torch.matmul(z_e_flat, self.embedding.weight.t()) # (N*H*W, K)
        
        # Distanza al quadrato: (N*H*W, 1) + (K,) - 2*(N*H*W, K) -> (N*H*W, K)
        distances = z_e_norm_sq + e_norm_sq - 2 * e_dot_z
        
        # 3. Trovare l'indice più vicino (argmin)
        # Questa è l'operazione q(z=k|x) = 1 se k = argmin(...)
        # `indices` contiene l'indice k (da 0 a K-1) per ogni vettore z_e.
        # indices ha forma (N*H*W,)
        indices = torch.argmin(distances, dim=1)
        
        # 4. Recuperare i vettori quantizzati (z_q)
        # Usiamo gli indici per creare il tensore quantizzato.
        # `indices.view(-1)` assicura che sia 1D
        z_q_flat = self.embedding(indices.view(-1)) # (N*H*W, D)
        
        # 5. Calcolo della VQ Loss (per addestrare il codebook)
        # Loss = || SG(z_e) - e_k ||^2
        # Questa loss spinge i vettori e_k del codebook verso gli z_e
        # dell'encoder. Usiamo detach() su z_e_flat per non propagare
        # gradienti all'encoder (come da appunti).
        vq_loss = F.mse_loss(z_e_flat.detach(), z_q_flat)
        
        # 5. Gradient Pass-Through (Stop-Gradient)
        # L'operazione argmin non è differenziabile. Copiamo i gradienti
        # da z_q a z_e usando il trucco "straight-through estimator".
        # z_q = z_e + SG(z_q - z_e)
        # SG è l'operatore stop-gradient (detach() in PyTorch)
        z_q_flat = z_e_flat + (z_q_flat - z_e_flat).detach()
        
        # Riformattiamo z_q per avere la stessa forma dell'input z_e (serve per il decoder)
        z_q = z_q_flat.view(N, H, W, D).permute(0, 3, 1, 2) # (N, D, H, W)
        
        # Ritorna z_q, la vq_loss e gli indici (utili per un prior)
        return z_q, vq_loss, indices.view(N, H, W)


# --- 2. Architettura VQ-VAE ---
# Mettiamo insieme Encoder, VectorQuantizer e Decoder
class VQVAE(nn.Module):
    def __init__(self, input_dim, D, K):
        super(VQVAE, self).__init__()
        
        # L'Encoder (es. CNN) produce z_e(x)
        # Qui usiamo un'architettura semplice per dati 2D (come la semi-circonferenza)
        # Per immagini, useremmo Conv2d
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, D) # Output z_e di dimensione D
            # N.B. Per immagini 2D (es. 128x128), l'encoder produrrebbe
            # (N, D, H, W), es. (16, 64, 16, 16).
            # Per semplicità qui, l'output sarà (N, D, 1, 1)
        )
        
        # Il Codebook
        self.quantizer = VectorQuantizer(K, D)
        
        # Il Decoder (simmetrico all'encoder)
        self.decoder = nn.Sequential(
            nn.Linear(D, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim)
        )
        
        self.D = D
        self.K = K

    def forward(self, x):
        # 1. Encoder
        z_e = self.encoder(x) # (N, D)
        
        # 2. Adattamento Forma per Quantizer
        # Il nostro quantizer si aspetta (N, D, H, W).
        # Aggiungiamo dimensioni fittizie H=1, W=1.
        z_e = z_e.unsqueeze(-1).unsqueeze(-1) # (N, D, 1, 1)
        
        # 3. Quantizzazione
        # z_q ha la forma (N, D, 1, 1)
        z_q, vq_loss, indices = self.quantizer(z_e)
        
        # 4. Adattamento Forma per Decoder
        # Rimuoviamo le dimensioni fittizie
        z_q = z_q.squeeze(-1).squeeze(-1) # (N, D)
        
        # 5. Decoder
        # Ricostruisce x da z_q
        x_recon = self.decoder(z_q)
        
        # Ritorna l'output, la loss del VQ e z_e (per la commitment loss)
        # z_e viene ritornato come (N, D)
        return x_recon, vq_loss, z_e.squeeze(-1).squeeze(-1)