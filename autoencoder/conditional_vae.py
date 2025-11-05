import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from cvae_model import ConditionalVAE # Importa il modello CVAE

# --- 1. VAE Loss Function (Identica al VAE standard) ---
def vae_loss_function(recon_x, x, mu, log_var, beta):
    """
    Calcola la loss del CVAE, che è identica a quella del VAE.
    L'obiettivo è sempre massimizzare l'ELBO.
    Loss = Reconstruction_Loss + KL_Divergence_Loss
    
    - Reconstruction Loss: -E[log p(x | z, c)]
      Misura quanto bene ricostruiamo x, dati z E c.
      
    - KL Divergence Loss: KL( q(z | x, c) || p(z) )
      Misura la "distanza" tra la nostra distribuzione encoder (ora
      condizionata anche a c) e il prior p(z).
      
    Il prior p(z) è ancora N(0, 1) e non dipende da c. Questo
    "costringe" l'encoder a mappare tutte le coppie (x, c) a uno
    spazio latente che, in aggregato, assomiglia al prior.
    """
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recon_loss + beta * kl_loss

# --- 2. Data Generation (Dati Condizionati) ---
# Creiamo un dataset dove l'input x dipende da un'etichetta c.
# Creeremo due "mezze lune" o cluster separati.
num_data_points_per_class = 2000
total_points = num_data_points_per_class * 2

# --- Classe 0 ---
# Una semi-circonferenza centrata a (-2, 0)
theta0 = torch.linspace(0, torch.pi, num_data_points_per_class)
x0_data = -2 + torch.cos(theta0)
y0_data = torch.sin(theta0)
class0_data = torch.stack([x0_data, y0_data], dim=1)
# Etichetta per la classe 0 (useremo one-hot encoding)
class0_labels = torch.zeros(num_data_points_per_class, 2)
class0_labels[:, 0] = 1.0

# --- Classe 1 ---
# Una semi-circonferenza centrata a (2, 0)
theta1 = torch.linspace(0, torch.pi, num_data_points_per_class)
x1_data = 2 + torch.cos(theta1)
y1_data = torch.sin(theta1)
class1_data = torch.stack([x1_data, y1_data], dim=1)
# Etichetta per la classe 1
class1_labels = torch.zeros(num_data_points_per_class, 2)
class1_labels[:, 1] = 1.0

# --- Combina e Aggiungi Rumore ---
# Dati (x)
data = torch.cat([class0_data, class1_data], dim=0)
data += torch.randn_like(data) * 0.05 # Aggiungi rumore
# Etichette (c)
labels = torch.cat([class0_labels, class1_labels], dim=0)
# Per la visualizzazione, teniamo traccia degli indici di classe (0 o 1)
label_indices = torch.cat([
    torch.zeros(num_data_points_per_class),
    torch.ones(num_data_points_per_class)
], dim=0)


# --- 3. Training the CVAE ---
# Definiamo le dimensioni
input_dim = 2       # Dimensione di x (x, c)
condition_dim = 2   # Dimensione di c (one-hot encoding [1,0] o [0,1])
latent_dim = 2      # Dimensione di z
learning_rate = 0.001
num_epochs = 2000
beta_value = 0.1


# Instanziamo il modello CVAE
model = ConditionalVAE(input_dim, condition_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

print(f"Training a Conditional VAE (CVAE) con {condition_dim} classi...")
# Training loop
for epoch in range(num_epochs):
    # --- Forward pass ---
    # Passiamo sia i dati 'data' (x) che le etichette 'labels' (c)
    reconstructions, mu, log_var = model(data, labels)
    
    # Calcoliamo la loss (la funzione è identica al VAE)
    loss = vae_loss_function(reconstructions, data, mu, log_var, beta=beta_value)

    # --- Backward pass e ottimizzazione ---
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item() / len(data):.4f}')

# --- 4. Visualization ---
model.eval()
with torch.no_grad():
    # --- 1. Ottieni Spazio Latente e Ricostruzioni ---
    # Dobbiamo passare i dati e le etichette originali
    reconstructed_data, mu, log_var = model(data, labels)
    z_latent_samples = model.reparameterize(mu, log_var)

    # --- 2. Generazione Condizionale (La parte più importante) ---
    # Per dimostrare che il condizionamento funziona, campioniamo z
    # dal PRIOR N(0, 1) e poi lo diamo al decoder insieme a
    # un'etichetta 'c' specifica.
    
    num_gen_samples = 500
    # Campiona z dal prior p(z) = N(0, 1)
    z_prior_samples = torch.randn(num_gen_samples, latent_dim)

    # --- Genera dati per la CLASSE 0 ---
    # Crea un tensore di etichette fittizie per la classe 0
    c_gen_0 = torch.zeros(num_gen_samples, condition_dim)
    c_gen_0[:, 0] = 1.0
    # Decodifica (z, c_0)
    generated_data_0 = model.decode(z_prior_samples, c_gen_0).numpy()

    # --- Genera dati per la CLASSE 1 ---
    # Crea un tensore di etichette fittizie per la classe 1
    c_gen_1 = torch.zeros(num_gen_samples, condition_dim)
    c_gen_1[:, 1] = 1.0
    # Decodifica (z, c_1)
    generated_data_1 = model.decode(z_prior_samples, c_gen_1).numpy()


# --- Creazione Plot ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
fig.suptitle('Conditional Variational Autoencoder (CVAE) Analysis', fontsize=16)

# --- Plot 1: Spazio Latente (Colorato per Classe) ---
# Mostra dove l'encoder q(z|x,c) mappa i dati.
# Dovremmo vedere le due classi mappate nello stesso spazio N(0,1).
scatter1 = ax1.scatter(
    z_latent_samples[:, 0], z_latent_samples[:, 1], c=label_indices, 
    cmap='coolwarm', s=10, alpha=0.7
)
ax1.set_title('1. Latent Space q(z|x,c)')
ax1.set_xlabel('Latent Dim 1 (z1)')
ax1.set_ylabel('Latent Dim 2 (z2)')
ax1.axis('equal')
ax1.grid(True)
ax1.legend(handles=scatter1.legend_elements()[0], labels=['Classe 0', 'Classe 1'])
ax1.text(0.5, -0.15,
         "Lo spazio latente è condiviso. La loss KL\nforza entrambe le classi nel prior N(0,1).",
         ha='center', transform=ax1.transAxes, style='italic')

# --- Plot 2: Ricostruzione ---
# Mostra i dati originali e le loro ricostruzioni
ax2.scatter(data[:, 0], data[:, 1], s=10, alpha=0.2, 
            c=label_indices, cmap='coolwarm', label='Original Data')
ax2.scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], s=20, 
            alpha=0.8, color='orange', marker='x',
            label='Reconstructed Data')
ax2.set_title('2. Reconstruction p(x|z,c)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.axis('equal')
ax2.legend()
ax2.grid(True)

# --- Plot 3: Generazione Condizionale ---
# Mostra i campioni generati dal prior p(z) e da c scelta.
ax3.scatter(data[:, 0], data[:, 1], s=10, alpha=0.1, 
            c=label_indices, cmap='coolwarm', label='Original Data (Riferimento)')
# Plotta i dati generati per la CLASSE 0
ax3.scatter(generated_data_0[:, 0], generated_data_0[:, 1], s=20, 
            alpha=0.8, color='blue', 
            label='Generated: Classe 0 (c=[1,0])')
# Plotta i dati generati per la CLASSE 1
ax3.scatter(generated_data_1[:, 0], generated_data_1[:, 1], s=20, 
            alpha=0.8, color='red',
            label='Generated: Classe 1 (c=[0,1])')
ax3.set_title('3. Conditional Generation p(x|z,c)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.axis('equal')
ax3.legend()
ax3.grid(True)
ax3.text(0.5, -0.15,
         "Campionando z~N(0,1) e fornendo una c specifica,\npossiamo controllare quale classe generare.",
         ha='center', transform=ax3.transAxes, style='italic', color='green')

# Mostra la figura finale
plt.tight_layout()
plt.show()