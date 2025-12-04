import torch
import sys
import os

# Aggiungiamo la root del progetto al path per importare i moduli se necessario
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimization.lora import LoRALinear

print("\n--- LoRA Analysis ---")
DIM_IN = 1024
DIM_OUT = 1024
RANK = 8

# Creiamo il layer
lora_layer = LoRALinear(DIM_IN, DIM_OUT, r=RANK)

# Contiamo i parametri
params_pretrained = sum(p.numel() for p in lora_layer.pretrained.parameters())
params_lora = sum(p.numel() for p in [lora_layer.lora_A, lora_layer.lora_B])
params_total = params_pretrained + params_lora

print(f"Layer standard (1024x1024): {params_pretrained:,} parametri")
print(f"LoRA Adapter (rank={RANK}):      {params_lora:,} parametri")
print(f"Incremento parametri: {params_lora / params_pretrained:.2%}")

# Test forward
x = torch.randn(1, DIM_IN)
y_initial = lora_layer(x)
print("Forward pass iniziale completato.")

# Simuliamo un update (cambiamo leggermente lora_B)
with torch.no_grad():
    lora_layer.lora_B.data += 0.01

y_updated = lora_layer(x)
diff = (y_updated - y_initial).abs().mean().item()
print(f"Differenza output dopo update di LoRA: {diff:.6f} (deve essere > 0)")