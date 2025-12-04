import torch
import sys
import os

# Aggiungiamo la root del progetto al path per importare i moduli se necessario
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from optimization.mla import MultiHeadLatentAttention

print("\n--- Multi-Head Latent Attention (MLA) vs Standard ---")
DIM = 2048
HEADS = 16 # Head dim = 128
SEQ_LEN = 1024
BATCH = 1

# Standard KV Cache Size (in elementi float)
# Batch * Seq * Heads * Head_Dim * 2 (K+V)
mha_cache_elements = BATCH * SEQ_LEN * HEADS * (DIM // HEADS) * 2
print(f"MHA Cache (Elementi): {mha_cache_elements:,}")

# MLA Configuration
LATENT_DIM_KV = 512 # Comprimiamo pesantemente KV (es. DeepSeek-V2 style)
mla = MultiHeadLatentAttention(DIM, HEADS, LATENT_DIM_KV)

# MLA KV Cache Size
# Batch * Seq * Latent_Dim_KV
# Nota: in MLA salviamo solo il vettore compresso!
mla_cache_elements = BATCH * SEQ_LEN * LATENT_DIM_KV
print(f"MLA Cache (Elementi): {mla_cache_elements:,}")

compression_ratio = mha_cache_elements / mla_cache_elements
print(f"Rapporto Compressione Memoria: {compression_ratio:.1f}x")

# Forward Pass Check
x_seq = torch.randn(BATCH, 10, DIM) # Sequenza corta per test
out = mla(x_seq)
print(f"MLA Output shape: {out.shape} (Match input: {x_seq.shape})")