import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.functional import binary_cross_entropy
from matplotlib import pyplot as plt
from math import pi
import matplotlib.gridspec as gridspec  # Importiamo GridSpec

"""
In this example, we will use a distribution of points in the 2-D plane as the distribution we want to generate using a GAN.
"""

"""The following code generates our training set. Later we will plot the points in
the training set to visualize the desired distribution.
"""
a=torch.rand(1600)*pi/4
r=torch.randn_like(a)*0.05+1.25
x=r*torch.cos(a)
y=r*torch.sin(a)
training_set=torch.stack((x,y), dim=1)
training_loader=DataLoader(training_set, batch_size=64, shuffle=True)

"""The following code defines our models and optimizers for the generator and for the discriminator. Note that we need separate optimizers."""
LATENT_SIZE=32
NUM_EPOCHS = 200

# Definiamo una funzione che esegue l'intero esperimento
# Modifichiamo gli argomenti per accettare assi separati
def run_experiment(generator_lr, num_epochs, epoch_axes, loss_ax):
    """
    Esegue un addestramento completo della GAN
    """
    
    print(f"\n--- Inizio Esperimento: Generatore LR = {generator_lr} ---")

    # 1. Definizione Modelli
    gen_model=nn.Sequential(
            nn.Linear(LATENT_SIZE, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2))

    disc_model=nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid())

    # 2. Definizione Ottimizzatori
    gen_optimizer=torch.optim.Adam(gen_model.parameters(), lr=generator_lr)
    disc_optimizer=torch.optim.Adam(disc_model.parameters())

    # 3. Funzioni di Loss
    def disc_loss_function(d_true, d_synth):
        t_true=torch.ones_like(d_true)
        t_synth=torch.zeros_like(d_synth)
        return binary_cross_entropy(d_true, t_true) + binary_cross_entropy(d_synth, t_synth)

    def gen_loss_function(d_synth):
        t_synth=torch.ones_like(d_synth)
        return binary_cross_entropy(d_synth, t_synth)

    # 4. Funzione di Training Epoch
    def training_epoch(dataloader):
        gen_model.train()
        disc_model.train()
        sum_gloss=0.0
        sum_dloss=0.0
        sum_dtrue=0.0
        sum_dsynth=0.0
        batches=0
        for x_true in dataloader:
            d_true=disc_model(x_true)
            z=torch.randn(x_true.shape[0], LATENT_SIZE)
            x_synth=gen_model(z)
            d_synth=disc_model(x_synth)

            disc_optimizer.zero_grad()
            dloss=disc_loss_function(d_true, d_synth)
            # se calcolassimo direttamente backward() perderemmo le informazioni sul grafo usato per la loss per il discriminator, 
            # viene fatto per motivi di ottimizzazione. 
            # Per evitare ciò e calcolare correttamente i gradienti per il generatore impostiamo `retain_graph=True`
            dloss.backward(retain_graph=True)
            disc_optimizer.step()

            # Nota che ricalcoliamo l’output del discriminatore rispetto ai dati generati prima di calcolare la loss del generatore, 
            # questo perché abbiamo aggiornato i pesi del discriminatore e quindi dobbiamo ricalcolare l’output del discriminatore
            d_synth=disc_model(x_synth)
            gen_optimizer.zero_grad()
            gloss=gen_loss_function(d_synth)
            gloss.backward()
            gen_optimizer.step()

            sum_gloss += gloss.item()
            sum_dloss += dloss.item()
            sum_dtrue += d_true.mean().item()
            sum_dsynth += d_synth.mean().item()
            batches += 1

            # Da notare che per l’addestramento di una GAN l’early stopping non serve, la loss è naturalmente instabile. 
            # Un criterio per decidere se fermarci è controllare che i campioni generati sono buoni

        print(f'GLoss: {sum_gloss/batches:.4f}',
              f'DLoss: {sum_dloss/batches:.4f}',
              f'Dtrue: {sum_dtrue/batches:.4f}',
              f'Dsynth: {sum_dsynth/batches:.4f}',
            )

        return sum_gloss/batches, sum_dloss/batches, sum_dtrue/batches, sum_dsynth/batches

    # 5. Funzioni di Plotting
    def show_discriminator(ax, disc_model):
        px=torch.linspace(-0.5, 2.0, 20)
        py=torch.linspace(-0.5, 2.0, 20)
        gridx, gridy=torch.meshgrid(px, py, indexing='ij')
        gx=gridx.reshape(-1)
        gy=gridy.reshape(-1)
        x=torch.stack((gx, gy), dim=1)
        disc_model.eval()
        with torch.no_grad():
            d=disc_model(x)
        d=d.reshape(gridx.shape)
        ax.pcolormesh(gridx, gridy, d, vmin=0.0, vmax=1.0, cmap='bwr_r', shading='gouraud')

    def show_generated(ax, gen_model, n):
        z=torch.randn(n, LATENT_SIZE)
        gen_model.eval()
        with torch.no_grad():
            x=gen_model(z)
        ax.scatter(x[:,0], x[:,1], c='red', edgecolor='black')

    def show_training_set(ax):
        ax.scatter(training_set[:,0], training_set[:,1], c='blue', edgecolor='black')

    def show_everything(ax, gen_model, disc_model):
        show_discriminator(ax, disc_model)
        show_training_set(ax)
        show_generated(ax, gen_model, 1000)
        ax.set_xlim(-0.5, 2.0)
        ax.set_ylim(-0.5, 2.0)
        ax.set_aspect('equal', adjustable='box')


    # 6. Esecuzione del Training Loop
    dlosses = []
    glosses = []
    dtrues = []
    dsynths = []
    
    # Gli indici per i plot intermedi (epoche 50, 100, 150, 200)
    plot_indices = {50: 0, 100: 1, 150: 2, 200: 3}
    
    print(f"Avvio addestramento per {num_epochs} epoche...")
    for i in range(num_epochs):
        epoch_num = i + 1
        print(f'[{i+1}]/[{num_epochs}]', end=' ')
        gloss, dloss, dtrue, dsynth = training_epoch(training_loader)
        dlosses.append(dloss)
        glosses.append(gloss)
        dtrues.append(dtrue)
        dsynths.append(dsynth)

        # Plotta sui subplot intermedi
        if epoch_num in plot_indices:
            plot_idx = plot_indices[epoch_num]
            # Usiamo epoch_axes, che è la lista di 4 assi
            current_ax = epoch_axes[plot_idx] 
            show_everything(current_ax, gen_model, disc_model)
            current_ax.set_title(f'Epoch {epoch_num} (Gen LR: {generator_lr})')
            

    # 7. Plot finale delle Loss sull'asse dedicato
    loss_ax.plot(dlosses, label='discriminator loss')
    loss_ax.plot(glosses, label='generator loss')
    loss_ax.plot(dtrues, label='discriminator true')
    loss_ax.plot(dsynths, label='discriminator synth')
    loss_ax.set_title(f'Andamento Loss (Gen LR: {generator_lr})')
    loss_ax.set_xlabel('epoche')
    loss_ax.set_ylabel('loss')
    loss_ax.legend()
    
    print(f"--- Fine Esperimento: Generatore LR = {generator_lr} ---")


# --- Creazione della Griglia e Esecuzione ---

print("Creazione della griglia di plot...")
# Creiamo una figura più alta per ospitare 3 righe
fig = plt.figure(figsize=(20, 15))

# Definiamo una griglia 3x4 (3 righe, 4 colonne)
gs = gridspec.GridSpec(3, 4, figure=fig)

# --- Assi per Esperimento 1 (LR 0.001) ---
# Riga 0: 4 plot per le epoche
epoch_axes_1 = [fig.add_subplot(gs[0, i]) for i in range(4)]
# Riga 2 (colonne 0 e 1): 1 plot per la loss, che occupa 2 celle
loss_ax_1 = fig.add_subplot(gs[2, 0:2]) 

# --- Assi per Esperimento 2 (LR 0.0002) ---
# Riga 1: 4 plot per le epoche
epoch_axes_2 = [fig.add_subplot(gs[1, i]) for i in range(4)]
# Riga 2 (colonne 2 e 3): 1 plot per la loss, che occupa le 2 celle rimanenti
loss_ax_2 = fig.add_subplot(gs[2, 2:4])

# Impostiamo titoli di riga (sul primo asse di ogni riga)
epoch_axes_1[0].set_ylabel('LR = 0.001\nEpochs', fontsize=16, fontweight='bold')
epoch_axes_2[0].set_ylabel('LR = 0.0002\nEpochs', fontsize=16, fontweight='bold')
loss_ax_1.set_ylabel('Loss Plots', fontsize=16, fontweight='bold')

# Esperimento 1: Generatore LR = 0.001
run_experiment(generator_lr=0.001, 
               num_epochs=NUM_EPOCHS, 
               epoch_axes=epoch_axes_1, 
               loss_ax=loss_ax_1)

# Esperimento 2: Generatore LR = 0.0002
run_experiment(generator_lr=0.0002, 
               num_epochs=NUM_EPOCHS, 
               epoch_axes=epoch_axes_2, 
               loss_ax=loss_ax_2)

# --- Visualizzazione di tutti i grafici ---
print("\nAddestramenti completati. Mostro la griglia dei grafici...")

# Aggiungiamo un titolo generale
fig.suptitle('Confronto Addestramenti GAN', fontsize=20, fontweight='bold')

# Ottimizziamo il layout per evitare sovrapposizioni. 
# 'rect' lascia spazio per il suptitle
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Mostra la figura finale
plt.show()