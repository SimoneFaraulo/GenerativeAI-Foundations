import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention
import math

class GroupedQueryAttention(nn.Module):
    """
    Implementa la Grouped-Query Attention (GQA) e Multi-Query Attention (MQA).
    
    Differenze rispetto alla Multi-Head Attention (MHA) standard:
    - MHA: Num_Heads_Q = Num_Heads_K = Num_Heads_V (1:1 mapping)
    - GQA: Num_Heads_Q > Num_Heads_K (N:1 mapping per gruppo). K e V sono condivisi tra gruppi di Q.
    - MQA: Num_Heads_K = 1 (Tutte le teste Q condividono un'unica testa K e V). È un caso speciale di GQA.
    
    Vantaggi:
    Riduce drasticamente la dimensione della KV Cache e la memory bandwidth necessaria 
    durante l'inferenza, mantenendo prestazioni simili a MHA.
    """
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads          # Teste per la Query
        self.num_kv_heads = num_kv_heads    # Teste per Key e Value
        
        # Verifica per GQA/MQA
        assert num_heads % num_kv_heads == 0, "Il numero di teste Q deve essere multiplo delle teste KV"
        
        self.head_dim = dim // num_heads
        
        # Quante volte dobbiamo ripetere K e V per allinearci con Q?
        # Esempio GQA: 8 teste Q, 2 teste KV -> n_rep = 4
        # Esempio MQA: 8 teste Q, 1 testa KV -> n_rep = 8
        self.n_rep = num_heads // num_kv_heads
        
        # Proiezione Query: dimensione piena (come in MHA)
        self.q_proj = nn.Linear(dim, num_heads * self.head_dim, bias=False)
        
        # Proiezioni Key e Value: dimensione ridotta!
        # Invece di proiettare su 'dim', proiettiamo su 'num_kv_heads * head_dim'
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """
        Ripete le teste K/V per farle combaciare con il numero di teste Q.
        Input: (N, Num_KV_Heads, L, Head_Dim)
        Output: (N, Num_Heads, L, Head_Dim)
        """
        if n_rep == 1:
            return x
            
        N, num_kv_heads, L, head_dim = x.shape
        
        # Espandiamo la dimensione delle teste: (N, num_kv_heads, 1, L, head_dim)
        x = x[:, :, None, :, :]
        # Ripetiamo lungo la nuova dimensione: (N, num_kv_heads, n_rep, L, head_dim)
        x = x.expand(N, num_kv_heads, n_rep, L, head_dim)
        # Appiattiamo per unire i gruppi: (N, num_kv_heads * n_rep, L, head_dim)
        # Dato che num_kv_heads * n_rep = num_heads, otteniamo la forma corretta per Q.
        return x.reshape(N, num_kv_heads * n_rep, L, head_dim)

    def forward(self, 
                x: torch.Tensor, 
                kv_cache = None, 
                is_causal: bool = False) -> torch.Tensor:
        """
        Input:
          x: (N, L, D) - Input corrente
          kv_cache: Oggetto KVCache opzionale per l'inferenza
        """
        N, L, _ = x.shape
        
        # 1. Proiezioni
        q = self.q_proj(x) # (N, L, num_heads * head_dim)
        k = self.k_proj(x) # (N, L, num_kv_heads * head_dim) -> Nota la dimensione ridotta
        v = self.v_proj(x) # (N, L, num_kv_heads * head_dim)

        # 2. Reshape in teste
        q = q.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)       # (N, H_q, L, D)
        k = k.view(N, L, self.num_kv_heads, self.head_dim).transpose(1, 2)    # (N, H_kv, L, D)
        v = v.view(N, L, self.num_kv_heads, self.head_dim).transpose(1, 2)    # (N, H_kv, L, D)
        
        # 3. Gestione KV Cache (Se presente, siamo in fase di generazione)
        if kv_cache is not None:
            # Aggiorna la cache e recupera la storia completa concatenata
            k, v = kv_cache.update(k, v)
            
        # 4. Ripetizione (Broadcasting) di K e V
        # Per calcolare l'attenzione, K e V devono avere lo stesso numero di teste di Q.
        # In MQA/GQA, replichiamo le teste esistenti.
        # (N, H_kv, ...) -> (N, H_q, ...)
        k = self.repeat_kv(k, self.n_rep)
        v = self.repeat_kv(v, self.n_rep)
        
        # 5. Scaled Dot Product Attention
        # PyTorch gestisce efficientemente il calcolo
        # is_causal applica la maschera triangolare se necessario
        output = scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal
        )
        
        # 6. Output Projection
        output = output.transpose(1, 2).contiguous().view(N, L, self.dim)
        return self.out_proj(output)

# --- Classi helper per chiarezza ---

class MultiQueryAttention(GroupedQueryAttention):
    """
    MQA è un caso speciale di GQA dove num_kv_heads = 1.
    Massima compressione della memoria.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(dim, num_heads, num_kv_heads=1, dropout=dropout)

class MultiHeadAttentionStandard(GroupedQueryAttention):
    """
    MHA standard è un caso speciale di GQA dove num_kv_heads = num_heads.
    Nessuna compressione.
    """
    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__(dim, num_heads, num_kv_heads=num_heads, dropout=dropout)