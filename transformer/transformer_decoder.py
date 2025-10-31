import torch
from torch import nn
from attention import MultiHeadAttention
from transformer_encoder import FeedForward # Riusiamo il modulo FFN

class DecoderLayer(nn.Module):
    """
    Implementa un singolo strato (layer) del Decoder.
    
    Ogni layer del decoder ha tre sotto-componenti principali:
    1. Un meccanismo di Multi-Head Self-Attention **Mascherato** (Masked).
       Questo assicura che un token alla posizione 'i' possa dipendere solo
       dai token precedenti (posizioni < i).
    2. Un meccanismo di Multi-Head **Cross-Attention**.
       Query(Q) proviene dal decoder (dal blocco self-attn precedente).
       Key(K) e Value(V) provengono dall'output dell'Encoder.
       Questo permette al decoder di "consultare" la sequenza di input.
    3. Una rete Feed-Forward (FFN), identica a quella dell'encoder.
    
    Anche qui, ogni sotto-componente è seguito da Add & Norm.
    """
    def __init__(self, dim: int, heads: int, hidden_dim_multiplier: int = 4, dropout: float = 0.1):
        super().__init__()
        
        # 1. Masked Multi-Head Self-Attention
        self.masked_self_attn = MultiHeadAttention(dim, heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # 2. Multi-Head Cross-Attention
        self.cross_attn = MultiHeadAttention(dim, heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout2 = nn.Dropout(dropout)
        
        # 3. Feed-Forward Network
        self.ffn = FeedForward(dim, dim * hidden_dim_multiplier, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, Z: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Input:
          X: (N, L_tgt, D) - Input dal decoder (sequenza target)
          Z: (N, L_src, D) - Output dell'encoder (sequenza sorgente)
          src_mask: (N, 1, L_src) - Maschera di padding per la sorgente (usata in cross-attn)
          tgt_mask: (N, L_tgt, L_tgt) - Maschera di padding per il target (usata in self-attn)
                                       N.B. la maschera causale è gestita da 'is_causal=True'
        """
        
        # --- Sotto-strato 1: Masked Self-Attention ---
        # Il decoder deve ignorare i token futuri.
        # 'is_causal=True' applica la maschera triangolare.
        # 'tgt_mask' (se presente) gestisce il padding della sequenza target.
        attn_output = self.masked_self_attn(
            query=X, 
            key=X, 
            value=X, 
            mask=tgt_mask, # Maschera per il padding
            is_causal=True # Maschera per la causalità
        )
        attn_output = self.dropout1(attn_output) # dropout
        X = self.norm1(X + attn_output) # add & norm
        
        # --- Sotto-strato 2: Cross-Attention ---
        # Q = X (dal decoder), K = Z (output encoder), V = Z (output encoder)
        # 'is_causal' è False: il decoder può guardare ovunque nell'input.
        # 'src_mask' (se presente) maschera il padding dell'input dell'encoder.
        cross_attn_output = self.cross_attn(
            query=X,
            key=Z, 
            value=Z, 
            mask=src_mask,
            is_causal=False
        )
        cross_attn_output = self.dropout2(cross_attn_output)
        X = self.norm2(X + cross_attn_output) # add & norm
        
        # --- Sotto-strato 3: Feed-Forward ---
        ffn_output = self.ffn(X)
        ffn_output = self.dropout3(ffn_output)
        X_tilde = self.norm3(X + ffn_output) # add & norm
                
        # X_tilde è diretto al layer decoder successivo
        return X_tilde 

class Decoder(nn.Module):
    """
    Implementa il Decoder completo del Transformer.
    È uno stack di N DecoderLayer identici.
    """
    def __init__(self, 
                 num_layers: int, 
                 dim: int, 
                 heads: int, 
                 hidden_dim_multiplier: int = 4, 
                 dropout: float = 0.1):
        
        super().__init__()
        self.layers = nn.ModuleList([
            DecoderLayer(dim, heads, hidden_dim_multiplier, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, X: torch.Tensor, Z: torch.Tensor, src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Input:
          X: (N, L_tgt, D) - Input del decoder (già embeddato e posizionato)
          Z: (N, L_src, D) - Output dell'encoder
          src_mask: Maschera di padding della sorgente
          tgt_mask: Maschera di padding del target
        """
        # Passa l'input attraverso tutti i layer
        for layer in self.layers:
            X = layer(X, Z, src_mask, tgt_mask)
            
        # Applica la normalizzazione finale
        Y = self.norm(X)
        return Y