import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from vq_vae_model import VQVAE
import numpy as np


# --- 1. Definizione della Funzione di Loss VQ-VAE ---
def vq_vae_loss_function(x_recon, x, vq_loss, z_e, z_q_detached, beta):
    """
    Calcola la loss totale del VQ-VAE come descritto negli appunti.
    
    Loss = Reconstruction_Loss + VQ_Loss + Commitment_Loss
    """
    
    # 1. Reconstruction Loss: -log p(x | z_q(x))
    # Misura quanto bene il decoder ricostruisce l'input originale x
    # partendo dal vettore quantizzato z_q.
    # Usiamo MSE come approssimazione (comune per p Gaussiana)
    recon_loss = nn.functional.mse_loss(x_recon, x)
    
    # 2. VQ Loss (Codebook Loss): || SG(z_e) - e_k ||^2
    # Questa loss è già calcolata all'interno del modulo VectorQuantizer
    # (passata come `vq_loss`).
    # Aggiorna solo i vettori e_k nel codebook per farli avvicinare
    # agli output z_e dell'encoder.
    
    # 3. Commitment Loss: beta * || z_e - SG(e_k) ||^2
    # Assicura che l'output dell'encoder z_e non diverga troppo dai
    # vettori del codebook e_k a cui è stato mappato.
    # Aggiorna solo i pesi dell'encoder.
    # `z_q_detached` è z_q (ovvero e_k) con stop-gradient (SG).
    commitment_loss = beta * nn.functional.mse_loss(z_e, z_q_detached)
    
    # Loss totale
    # La VQ_Loss è già `vq_loss`.
    total_loss = recon_loss + vq_loss + commitment_loss
    
    return total_loss, recon_loss, commitment_loss

# --- 2. Data Generation (Semi-circle Manifold) ---
num_data_points = 2000
theta = torch.linspace(0, torch.pi, num_data_points)
x_data = torch.cos(theta)
y_data = torch.sin(theta)
data = torch.stack([x_data, y_data], dim=1)
data += torch.randn_like(data) * 0.05

# --- 3. Training the VQ-VAE ---
# Definiamo le dimensioni
input_dim = 2   # Il nostro dato è 2D (x, y)
latent_dim_D = 4 # Dimensione D dei vettori nel codebook
num_embeddings_K = 5 # Numero K di vettori nel codebook

# Iperparametri
learning_rate = 0.001
num_epochs = 2000
beta = 0.25 # Peso della Commitment Loss (come da appunti)

# Instanziamo il modello
model = VQVAE(input_dim, latent_dim_D, num_embeddings_K)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Training VQ-VAE (K={num_embeddings_K}, D={latent_dim_D})...")
for epoch in range(num_epochs):
    
    # --- Forward pass ---
    # N.B. Dobbiamo recuperare z_q_detached per la commitment loss.
    # Per farlo, ri-calcoliamo z_q ma con .detach() su z_e.
    
    # 1. Passaggio standard
    x_recon, vq_loss, z_e = model(data)
    
    # 2. Calcolo di z_q_detached (SG(e_k))
    # Passiamo z_e attraverso il quantizer, ma prima applichiamo detach()
    # a z_e per assicurarci che i gradienti non fluiscano a z_e.
    # Questo isola e_k come SG(e_k).
    z_e_sg = z_e.detach().unsqueeze(-1).unsqueeze(-1) # (N, D, 1, 1)
    z_q_detached, _, _ = model.quantizer(z_e_sg)
    z_q_detached = z_q_detached.squeeze(-1).squeeze(-1) # (N, D)

    # 3. Calcolo della Loss totale
    loss, recon_l, commit_l = vq_vae_loss_function(
        x_recon, data, vq_loss, z_e, z_q_detached, beta
    )

    # --- Backward pass e ottimizzazione ---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f} '
              f'(Recon: {recon_l.item():.4f}, VQ: {vq_loss.item():.4f}, Commit: {commit_l.item():.4f})')

# --- 4. Visualization ---
model.eval()
with torch.no_grad():
    # Otteniamo le ricostruzioni
    reconstructed_data, _, _ = model(data)
    
    # Decodifichiamo TUTTI i K vettori del codebook per vedere gli "archetipi"
    # model.quantizer.embedding.weight è il tensore (K, D)
    all_codebook_vectors = model.quantizer.embedding.weight 
    # Passiamoli nel decoder
    decoded_archetypes = model.decoder(all_codebook_vectors).numpy()
    
    # Troviamo gli indici k usati per i dati originali
    z_e_final = model.encoder(data).unsqueeze(-1).unsqueeze(-1)
    _, _, indices = model.quantizer(z_e_final) # (N, 1, 1)
    indices = indices.squeeze().numpy()
    
    # Controlliamo quanti codici unici sono stati usati
    unique_indices = np.unique(indices)
    print(f"Indici del codebook utilizzati (su {num_embeddings_K}): {len(unique_indices)}")
    print(f"Indici specifici: {unique_indices}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Vector Quantization VAE (VQ-VAE)', fontsize=16)

# --- PLOT 1 ---
# Plot 1: Ricostruzione e Archetipi Appresi
# Original Data: Punti 'o' blu, trasparenti
ax1.scatter(data[:, 0], data[:, 1], 
            s=20, alpha=0.25, color='blue', marker='o', 
            label='Original Data', zorder=1)
# Reconstructed Data: Punti 'x' arancioni, più opachi
ax1.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], 
            s=150, alpha=1.0, color='darkorange', marker='x', edgecolor='black', linewidth=3.0,
            label='Reconstructed Data', zorder=2)
# Archetypes: Stelle rosse
ax1.scatter(decoded_archetypes[:, 0], decoded_archetypes[:, 1], 
            s=150, marker='*', color='red', edgecolor='black', 
            label=f'Learned Archetypes (K={num_embeddings_K})',
            zorder=3)
ax1.set_title('1. Reconstruction and Learned Archetypes')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.axis('equal')
ax1.legend()
ax1.grid(True)
ax1.text(0.5, -0.15,
         "Gli 'Archetipi' (stelle) sono i punti generati decodificando\nognuno dei K vettori del codebook.",
         ha='center', transform=ax1.transAxes, style='italic')

# --- PLOT 2  ---
# Plot 2: Spazio Latente (Colorato con mappa di colori qualitativa)
# Creiamo una mappa di colori discreta con K colori
# Usiamo 'tab20' o 'Paired' che sono buone per i colori qualitativi
base_cmap = plt.cm.get_cmap('Paired', num_embeddings_K)
cmap = mcolors.ListedColormap(base_cmap(range(num_embeddings_K)))
# Definiamo i bordi per la colorbar
bounds = np.arange(num_embeddings_K + 1) - 0.5
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Plottiamo lo scatter plot
scatter = ax2.scatter(data[:, 0], data[:, 1], c=indices, cmap=cmap, norm=norm, s=15, alpha=0.9)
ax2.set_title(f'2. Latent Space (Data colored by {num_embeddings_K} discrete codes)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.axis('equal')
ax2.grid(True)
ax2.text(0.5, -0.15,
         f"I colori mostrano a quale dei K indici ogni punto è mappato.\nFenomeno 'Codebook Collapse': solo {len(unique_indices)} colori/indici sono usati.",
         ha='center', transform=ax2.transAxes, style='italic', color='red')

# Aggiungiamo la Colorbar come legenda
# Definiamo i tick per mostrare gli indici discreti
ticks = np.arange(num_embeddings_K)
cbar = fig.colorbar(scatter, ax=ax2, cmap=cmap, norm=norm, boundaries=bounds, ticks=ticks)
cbar.set_label('Indice del Codebook (k)')

plt.tight_layout()
plt.show()