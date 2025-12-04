import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

class MultiHeadLatentAttention(nn.Module):
    """
    Implementa la Multi-Head Latent Attention (MLA).
    
    Il problema di MHA/GQA:
    Dobbiamo memorizzare grosse matrici K e V nella cache per ogni token.
    Dimensione Cache MHA ~= Batch * Seq * (Num_Heads * Head_Dim)
    
    La soluzione MLA:
    Invece di memorizzare le teste K e V complete, memorizziamo un vettore latente compresso.
    
    Struttura:
    1. Input -> Down-Projection -> Latent Vector (Compresso, dimensione molto piccola)
    2. Latent Vector -> Up-Projection -> Heads K e Heads V (Decompresso al volo)
    
    In inferenza, grazie alle proprietà dell'algebra lineare, possiamo assorbire le matrici 
    di Up-Projection nella matrice di Query, permettendoci di non decomprimere mai K e V 
    in memoria, riducendo drasticamente il consumo di VRAM.
    """
    def __init__(self, dim: int, num_heads: int, latent_dim_kv: int, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        # Dimensione del vettore latente compresso (molto minore di dim!)
        self.latent_dim_kv = latent_dim_kv 
        
        # --- Proiezioni Query (Standard) ---
        self.q_proj = nn.Linear(dim, dim, bias=False)
        
        # --- Compressione KV (MLA) ---
        # 1. Down-projection: Comprime l'input in un vettore latente
        self.kv_down_proj = nn.Linear(dim, latent_dim_kv, bias=False)
        self.norm_kv = nn.LayerNorm(latent_dim_kv) # Normalizzazione importante per la stabilità
        
        # 2. Up-projections: Genera K e V dal vettore latente
        # Generiamo le teste "virtuali" per K e V
        self.kv_up_proj_k = nn.Linear(latent_dim_kv, dim, bias=False)
        self.kv_up_proj_v = nn.Linear(latent_dim_kv, dim, bias=False)
        
        # --- Output ---
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        """
        Input: (N, L, D)
        """
        N, L, _ = x.shape
        
        # 1. Calcolo Query (Standard)
        q = self.q_proj(x)
        q = q.view(N, L, self.num_heads, self.head_dim).transpose(1, 2) # (N, H, L, head_dim)
        
        # 2. Calcolo KV Compresso (MLA)
        # Proiettiamo verso il basso nello spazio latente
        c_kv = self.kv_down_proj(x) # (N, L, latent_dim_kv)
        c_kv = self.norm_kv(c_kv)
        
        # Nota:
        # In fase di inferenza pura, salveremmo nella cache SOLO 'c_kv'.
        # Risparmio memoria = latent_dim_kv / (num_heads * head_dim)
        # Esempio: latent=512, full=4096 -> 8x compressione (migliore anche di GQA 8:1 perché comprime anche V)
        
        # 3. Decompressione (Generazione on-the-fly delle teste)
        k = self.kv_up_proj_k(c_kv) # (N, L, dim)
        v = self.kv_up_proj_v(c_kv) # (N, L, dim)
        
        # Reshape in teste standard
        k = k.view(N, L, self.num_heads, self.head_dim).transpose(1, 2) # (N, H, L, head_dim)
        v = v.view(N, L, self.num_heads, self.head_dim).transpose(1, 2) # (N, H, L, head_dim)
        
        # 4. Attention (Standard)
        # Una volta decompresso, il calcolo è identico a MHA
        output = scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal
        )
        
        output = output.transpose(1, 2).contiguous().view(N, L, self.dim)
        return self.out_proj(output)