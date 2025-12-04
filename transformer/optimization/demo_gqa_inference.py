import torch
import time
from kv_cache import KVCache
from attention_variants import MultiHeadAttentionStandard, MultiQueryAttention, GroupedQueryAttention

# --- Configurazione ---
BATCH_SIZE = 1
SEQ_LEN = 10     # Lunghezza del prompt iniziale
GEN_LEN = 5      # Token da generare
DIM = 512
HEADS = 8
DEVICE = torch.device("cpu")

print("--- 1. Confronto Parametri ---")

# 1. Standard MHA (8 teste Q, 8 teste KV)
mha = MultiHeadAttentionStandard(DIM, HEADS).to(DEVICE)
mha_params = sum(p.numel() for p in mha.parameters())

# 2. GQA (8 teste Q, 2 teste KV) -> Compressione 4x di K/V
gqa = GroupedQueryAttention(DIM, HEADS, num_kv_heads=2).to(DEVICE)
gqa_params = sum(p.numel() for p in gqa.parameters())

# 3. MQA (8 teste Q, 1 testa KV) -> Compressione 8x di K/V
mqa = MultiQueryAttention(DIM, HEADS).to(DEVICE)
mqa_params = sum(p.numel() for p in mqa.parameters())

print(f"MHA Params: {mha_params:,} (Baseline)")
print(f"GQA Params: {gqa_params:,} (Riduzione KV Heads da 8 a 2)")
print(f"MQA Params: {mqa_params:,} (Riduzione KV Heads da 8 a 1)")
print("Nota come MQA/GQA riducono i parametri nelle proiezioni K e V.")

# --- Simulazione Generazione con KV Cache ---
print("\n--- 2. Simulazione Generazione Autoregressiva (GQA + KV Cache) ---")

# Creiamo input fittizio (Prompt)
x = torch.randn(BATCH_SIZE, SEQ_LEN, DIM).to(DEVICE)

# Inizializziamo la cache
head_dim = DIM // HEADS
# Nota: La cache deve essere dimensionata per le KV heads (2), non per le Q heads (8)
cache = KVCache(max_seq_len=SEQ_LEN + GEN_LEN, 
                batch_size=BATCH_SIZE, 
                num_heads=2,  # GQA usa 2 teste KV
                head_dim=head_dim, 
                device=DEVICE)

print(f"Input Prompt shape: {x.shape}")

# Fase 1: Prefill (Processiamo tutto il prompt)
# In questa fase la cache viene riempita con i primi 10 token
with torch.no_grad():
    output = gqa(x, kv_cache=cache, is_causal=True)

print(f"Output Prefill shape: {output.shape}")
print(f"Stato Cache dopo Prefill: {cache.current_seq_len} tokens")

# Fase 2: Decoding (Generazione token-by-token)
print("Inizio generazione token-by-token...")
current_token = output[:, -1:, :] # Prendiamo l'ultimo vettore come "nuovo token"

for i in range(GEN_LEN):
    with torch.no_grad():
        # Passiamo SOLO l'ultimo token al modello, non tutta la sequenza!
        # La cache fornirà il contesto passato.
        # Input shape: (1, 1, 512)
        next_vector = gqa(current_token, kv_cache=cache, is_causal=True)
        
        # In un modello reale qui ci sarebbe: Logits -> Softmax -> Sampling -> Embedding
        # Qui usiamo semplicemente l'output come input per il prossimo step
        current_token = next_vector
        
    print(f"Step {i+1}: Generato token. Cache size: {cache.current_seq_len}")

print("\n--- Verifica Integrità Cache ---")
k_stored, v_stored = cache.k_cache, cache.v_cache
print(f"Cache K finale shape: {k_stored.shape}")
print(f"Cache V finale shape: {v_stored.shape}")
print(f"Deve essere: (Batch, KV_Heads=2, Total_Len=15, Head_Dim={head_dim})")

assert k_stored.shape == (BATCH_SIZE, 2, SEQ_LEN + GEN_LEN, head_dim)
print("Test Superato: La cache ha la dimensione corretta per GQA.")