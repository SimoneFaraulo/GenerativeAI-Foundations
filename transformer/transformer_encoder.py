import torch
from torch import nn
from attention import MultiHeadAttention

class FeedForward(nn.Module):
    """
    Implementa il blocco Feed-Forward (FFN) del Transformer.
    
    Questo blocco viene applicato indipendentemente a ogni posizione (token)
    nella sequenza. Consiste in due trasformazioni lineari con un'attivazione
    ReLU (o GELU) nel mezzo.
    
    FFN(x) = max(0, xW1 + b1)W2 + b2
    
    Solitamente la dimensione interna (d_ff) è 4 volte la dimensione
    del modello (dim).
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    """
    Implementa un singolo strato (layer) dell'Encoder.
    
    Ogni layer dell'encoder ha due sotto-componenti principali:
    1. Un meccanismo di Multi-Head Self-Attention.
    2. Una rete Feed-Forward (FFN) semplice e posizionale.
    
    Intorno a ciascuno di questi due componenti c'è una connessione
    residua (residual connection) seguita da una normalizzazione
    (Layer Normalization).
    
    Struttura:
    x -> Self-Attention -> Add & Norm -> Feed-Forward -> Add & Norm -> output
    """
    def __init__(self, dim: int, heads: int, hidden_dim_multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # 1. Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(dim, heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # 2. Feed-Forward Network
        self.ffn = FeedForward(dim, dim * hidden_dim_multiplier, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Input:
          X: (N, L, D) - Input dalla sequenza
          mask: (N, 1, L) o (N, L, L) - Maschera di padding (per ignorare i token <PAD>)
        """
        
        # --- Sotto-strato 1: Self-Attention ---
        
        # Calcola l'attention. Nell'encoder, Q, K, V sono tutti 'X'.
        # 'is_causal' è False perché l'encoder può guardare l'intera sequenza.
        attn_output = self.self_attn(
            query=X, 
            key=X, 
            value=X, 
            mask=mask,
            is_causal=False 
        )
        
        # Connessione residua e normalizzazione (Pre-Norm o Post-Norm)
        # Qui usiamo Post-Norm (come nel paper originale):
        attn_output = self.dropout1(attn_output) # dropout
        Z = self.norm1(X + attn_output) # add & norm

        # --- Sotto-strato 2: Feed-Forward ---
        
        ffn_output = self.ffn(Z)
        X_tilde = self.norm2(Z + self.dropout2(ffn_output)) # add & norm
        
        # X_tilde è diretto al layer encoder successivo
        return X_tilde

class Encoder(nn.Module):
    """
    Implementa l'Encoder completo del Transformer.
    È uno stack di N EncoderLayer identici.
    """
    def __init__(self, 
                 num_layers: int, 
                 dim: int, 
                 heads: int, 
                 hidden_dim_multiplier: int = 4, 
                 dropout: float = 0.1):
        
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(dim, heads, hidden_dim_multiplier, dropout)
            for _ in range(num_layers)
        ])
        # Normalizzazione finale opzionale (usata in alcune implementazioni)
        self.norm = nn.LayerNorm(dim)

    def forward(self, X: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Input:
          X: (N, L, D) - Input (già passato attraverso l'embedding e il pos_encoding)
          mask: Maschera di padding
        """
        # Passa l'input attraverso tutti i layer
        for layer in self.layers:
            X = layer(X, mask)
            
        # Applica la normalizzazione finale
        Y = self.norm(X)
        return Y