import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.functional import binary_cross_entropy
from cgan_model import Generator, Discriminator # Importa i modelli

# --- 1. Definizione dei Parametri ---
LATENT_DIM = 16
CONDITION_DIM = 1
DATA_DIM = 2 # I nostri dati sono punti (x, y)
NUM_CLASSES = 2
LR = 0.0002
NUM_EPOCHS = 8000
BATCH_SIZE = 128
NOISE_STD = 0.25 # Deviazione standard dei cluster di dati reali

# --- OSTANTI PER LA STABILIZZAZIONE ---
# Rende la funzione del discriminatore più fluida e aiuta a guidare il generatore verso la regione dello spazio occupata dai campioni veri
LABEL_SMOOTHING = 0.1 
# Deviazione standard per il rumore sui campioni, questo rende il compito del discriminatore più difficile (in senso buono) 
# e garantisce che ci sia sempre un gradiente utile per il generatore.
INSTANCE_NOISE_STD = 0.05 

# Definiamo i centri dei nostri cluster
CENTERS = {
    0: torch.tensor([-1.0, -1.0]),
    1: torch.tensor([1.0, 1.0])
}

# --- 2. Funzione di Generazione Dati ---
def get_real_batch(batch_size):
    """
    Genera un batch di dati reali campionando casualmente da entrambe le classi.
    """
    c_int = torch.randint(0, NUM_CLASSES, (batch_size,))
    x_list = []
    c_list = []
    
    for i in c_int:
        class_idx = i.item()
        sample = CENTERS[class_idx] + torch.randn(DATA_DIM) * NOISE_STD
        x_list.append(sample)
        c_list.append(torch.tensor([float(class_idx)]))

    x_real = torch.stack(x_list)
    c_real = torch.stack(c_list)
    
    return x_real, c_real

def get_fake_batch(batch_size):
    """
    Genera un batch di input (z, c) per il generatore.
    """
    z = torch.randn(batch_size, LATENT_DIM)
    c_int = torch.randint(0, NUM_CLASSES, (batch_size,))
    c_fake = c_int.float().unsqueeze(1) 
    
    return z, c_fake

# --- 3. Definizione Modelli e Ottimizzatori ---
gen_model = Generator(LATENT_DIM, CONDITION_DIM, DATA_DIM)
disc_model = Discriminator(DATA_DIM, CONDITION_DIM)

gen_optimizer = optim.Adam(gen_model.parameters(), lr=LR, betas=(0.5, 0.999))
disc_optimizer = optim.Adam(disc_model.parameters(), lr=LR, betas=(0.5, 0.999))

# --- 4. Definizione Funzioni di Loss (MODIFICATE) ---
def disc_loss_function(d_true, d_synth):
    """Loss del discriminatore con Label Smoothing"""
    # Etichette "morbide" (soft labels)
    t_true = torch.ones_like(d_true) - LABEL_SMOOTHING
    t_synth = torch.zeros_like(d_synth) + LABEL_SMOOTHING
    return binary_cross_entropy(d_true, t_true) + binary_cross_entropy(d_synth, t_synth)

def gen_loss_function(d_synth):
    """Loss del generatore"""
    t_synth = torch.ones_like(d_synth)
    return binary_cross_entropy(d_synth, t_synth)

# --- 5. Training Loop ---
print("Inizio addestramento CGAN con Label Smoothing e Instance Noise...")
d_losses = []
g_losses = []

for epoch in range(NUM_EPOCHS):
    
    # --- 1. Addestramento del Discriminatore ---
    disc_optimizer.zero_grad()
    
    # Dati Reali
    x_real, c_real = get_real_batch(BATCH_SIZE)
    # --- RUMORE SUI CAMPIONI REALI ---
    x_real_noisy = x_real + torch.randn_like(x_real) * INSTANCE_NOISE_STD
    d_true = disc_model(x_real_noisy, c_real)
    
    # Dati Sintetici (Falsi)
    z, c_fake = get_fake_batch(BATCH_SIZE)
    x_synth = gen_model(z, c_fake).detach()
    # --- RUMORE SUI CAMPIONI FALSI ---
    x_synth_noisy = x_synth + torch.randn_like(x_synth) * INSTANCE_NOISE_STD
    d_synth = disc_model(x_synth_noisy, c_fake)
    
    # Calcolo Loss e Backpropagation
    dloss = disc_loss_function(d_true, d_synth)
    dloss.backward()
    disc_optimizer.step()

    # --- 2. Addestramento del Generatore ---
    gen_optimizer.zero_grad()
    
    z, c_fake = get_fake_batch(BATCH_SIZE)
    x_synth = gen_model(z, c_fake)
    
    # --- RUMORE SUI CAMPIONI FALSI PER IL GENERATORE ---
    # Il generatore deve imparare a ingannare un discriminatore
    # che si aspetta input rumorosi.
    x_synth_noisy_for_gen = x_synth + torch.randn_like(x_synth) * INSTANCE_NOISE_STD
    d_synth_for_gen = disc_model(x_synth_noisy_for_gen, c_fake)
    
    # Calcolo Loss e Backpropagation
    gloss = gen_loss_function(d_synth_for_gen)
    gloss.backward()
    gen_optimizer.step()

    # --- Logging ---
    d_losses.append(dloss.item())
    g_losses.append(gloss.item())
    
    if (epoch + 1) % 1000 == 0:
        print(f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss D: {dloss.item():.4f}, Loss G: {gloss.item():.4f}')

print("Addestramento completato.")

# --- 6. Visualizzazione (Invariata) ---

print("Visualizzazione dei risultati...")
fig, axs = plt.subplots(2, 2, figsize=(14, 14))
fig.suptitle('Analisi Conditional GAN (CGAN) con Stabilizzazione', fontsize=16)

# (Il codice di plotting da qui in poi è identico a prima)
# --- Plot 1: Dati Reali ---
ax = axs[0, 0]
x_real, c_real = get_real_batch(BATCH_SIZE * 2) # Ricampiona per pulizia
ax.scatter(x_real[:, 0], x_real[:, 1], c=c_real[:,0], cmap='coolwarm', alpha=0.6, label='Dati Reali')
ax.set_title('1. Dati Reali (Colorati per Classe)')
ax.legend()
ax.grid(True)

# --- Plot 2: Dati Generati ---
ax = axs[0, 1]
z, c_fake = get_fake_batch(BATCH_SIZE * 2)
x_synth = gen_model(z, c_fake).detach()
ax.scatter(x_synth[:, 0], x_synth[:, 1], c=c_fake[:,0], cmap='coolwarm', alpha=0.6, marker='x', label='Dati Generati')
ax.set_title('2. Dati Generati (Colorati per Classe)')
ax.legend()
ax.grid(True)

# --- Funzione helper per plottare la superficie di decisione del Discriminatore ---
def plot_decision_boundary(ax, class_idx):
    ax.set_title(f'3. Decisione D(x | c={class_idx})')
    px = torch.linspace(-4, 4, 50)
    py = torch.linspace(-4, 4, 50)
    gridx, gridy = torch.meshgrid(px, py, indexing='ij')
    gx = gridx.reshape(-1)
    gy = gridy.reshape(-1)
    x_grid = torch.stack((gx, gy), dim=1)
    c_grid = torch.full((x_grid.shape[0], 1), float(class_idx))
    
    disc_model.eval()
    with torch.no_grad():
        d = disc_model(x_grid, c_grid)
    d = d.reshape(gridx.shape)
    
    im = ax.pcolormesh(gridx, gridy, d, vmin=0.0, vmax=1.0, cmap='bwr_r', shading='gouraud', alpha=0.7)
    plt.colorbar(im, ax=ax, label='Probabilità "Reale"')
    ax.grid(True)

# --- Plot 3: Decisione D(x | c=0) ---
ax = axs[1, 0]
plot_decision_boundary(ax, 0)
x_real_c0, c_real_c0 = get_real_batch(200)
x_real_c0 = x_real_c0[c_real_c0[:,0] == 0]
ax.scatter(x_real_c0[:, 0], x_real_c0[:, 1], c='blue', edgecolor='black', label='Reali c=0')
z, c_fake_c0 = get_fake_batch(200)
c_fake_c0.fill_(0.0)
x_synth_c0 = gen_model(z, c_fake_c0).detach()
ax.scatter(x_synth_c0[:, 0], x_synth_c0[:, 1], c='red', marker='x', label='Sintetici c=0')
ax.legend()

# --- Plot 4: Decisione D(x | c=1) ---
ax = axs[1, 1]
plot_decision_boundary(ax, 1)
x_real_c1, c_real_c1 = get_real_batch(200)
x_real_c1 = x_real_c1[c_real_c1[:,0] == 1]
ax.scatter(x_real_c1[:, 0], x_real_c1[:, 1], c='blue', edgecolor='black', label='Reali c=1')
z, c_fake_c1 = get_fake_batch(200)
c_fake_c1.fill_(1.0)
x_synth_c1 = gen_model(z, c_fake_c1).detach()
ax.scatter(x_synth_c1[:, 0], x_synth_c1[:, 1], c='red', marker='x', label='Sintetici c=1')
ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- Plot delle Loss ---
plt.figure(figsize=(10, 5))
plt.plot(d_losses, label='Discriminator Loss')
plt.plot(g_losses, label='Generator Loss')
plt.title('Andamento delle Loss (Con Stabilizzazione)')
plt.xlabel('Epoca')
plt.ylabel('Loss')
plt.legend()
plt.show()