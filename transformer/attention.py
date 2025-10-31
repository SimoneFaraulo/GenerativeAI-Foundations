import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    """
    Implementa il layer di Multi-Head Attention.
    
    L'idea è di eseguire l'attention mechanism in parallelo più volte (con 'heads'
    differenti). Ogni "testa" si concentra su diverse parti/sottospazi
    dell'embedding.
    
    L'input (Query, Key, Value) viene proiettato linearmente 'heads' volte
    con diverse matrici di pesi. L'attention viene calcolata su queste
    proiezioni in parallelo. I risultati vengono poi concatenati e
    proiettati di nuovo per ottenere l'output finale.
    
    Questo modulo è generico:
    1. Per la Self-Attention: query=x, key=x, value=x
    2. Per la Cross-Attention: query=x_decoder, key=y_encoder, value=y_encoder
    """
    def __init__(self, dim: int, heads: int, dropout: float = 0.1):
        super().__init__()
        assert dim % heads == 0, "La dimensione (dim) deve essere divisibile per il numero di teste (heads)"
        
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads # Dimensione di ogni testa
        
        # Proiezioni lineari per Q, K, V e l'output
        # Usiamo un unico layer nn.Linear per efficienza invece di 3 separati
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        self.out_proj = nn.Linear(dim, dim) # Proiezione di output
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, X: torch.Tensor) -> torch.Tensor:
        """
        Divide l'ultima dimensione (dim) in (heads, head_dim) e
        riordina il tensore per l'elaborazione parallela delle teste.
        
        Input: (N, L, D) dove D = dim
        Output: (N, H, L, d_h) dove H = heads, d_h = head_dim
        """
        N, L, D = X.shape # (Batch_size, Seq_Len, Dim)
        X = X.reshape(N, L, self.heads, self.head_dim)
        return X.permute(0, 2, 1, 3) # (N, H, L, d_h)

    def join_heads(self, X: torch.Tensor) -> torch.Tensor:
        """
        Inverte l'operazione di split_heads.
        
        Input: (N, H, L, d_h)
        Output: (N, L, D) dove D = H * d_h
        """
        N, H, L, d_h = X.shape
        X = X.permute(0, 2, 1, 3) # (N, L, H, d_h)
        return X.reshape(N, L, self.dim) # (N, L, D)

    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor, 
                mask: torch.Tensor = None,
                is_causal: bool = False) -> torch.Tensor:
        """
        Input:
          query: (N, L_q, D) - Es. i token del decoder
          key:   (N, L_k, D) - Es. i token dell'encoder
          value: (N, L_k, D) - Es. i token dell'encoder
          mask: (N, L_q, L_k) o (L_q, L_k) - Maschera di padding (opzionale)
          is_causal: bool - Se True, applica una maschera causale (per decoder self-attention)
        """
        
        # 1. Proietta Q, K, V
        Q = self.q_proj(query) # (N, L_q, D)
        K = self.k_proj(key)   # (N, L_k, D)
        V = self.v_proj(value) # (N, L_k, D)
        
        # 2. Suddividi in teste
        Q = self.split_heads(Q) # (N, H, L_q, d_h)
        K = self.split_heads(K) # (N, H, L_k, d_h)
        V = self.split_heads(V) # (N, H, L_k, d_h)
        
        # 3. Calcola l'attention
        # scaled_dot_product_attention è un'implementazione ottimizzata
        # di PyTorch che fa:
        #   scores = (Q @ K.transpose(-2, -1)) / sqrt(d_h)
        #   scores = scores.masked_fill(mask == 0, -inf)
        #   attn_weights = softmax(scores, dim=-1)
        #   output = attn_weights @ V
        
        # 'attn_mask' e 'is_causal' sono mutuamente esclusivi in questa
        # funzione. Se 'is_causal' è True, la funzione crea internamente
        # una maschera triangolare. Se abbiamo una maschera di padding
        # (passata come 'mask'), dobbiamo usarla qui.
        
        if mask is not None:
             # Assicuriamoci che la maschera sia compatibile con il broadcasting
             # (N, H, L_q, L_k) o (N, 1, L_q, L_k)
             # La nostra maschera in input è (N, L_q, L_k) o (L_q, L_k)
             # Aggiungiamo la dimensione H
             if mask.dim() == 2: # (L_q, L_k)
                 mask = mask.unsqueeze(0).unsqueeze(0) # (1, 1, L_q, L_k)
             elif mask.dim() == 3: # (N, L_q, L_k)
                 mask = mask.unsqueeze(1) # (N, 1, L_q, L_k)
        
        # Nota: in PyTorch < 2.0, 'is_causal=True' e 'attn_mask' non possono
        # essere usati insieme. Nelle versioni moderne, 'is_causal' ha
        # la precedenza. Per la massima chiarezza, gestiamo 'is_causal'
        # separatamente.
        
        H = scaled_dot_product_attention(
            Q, K, V, 
            attn_mask=mask, 
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=is_causal
        ) # (N, H, L_q, d_h)
        
        # 4. Concatena le teste e proietta
        Y = self.join_heads(H) # (N, L_q, D)
        output = self.out_proj(Y) # (N, L_q, D)
        
        return output