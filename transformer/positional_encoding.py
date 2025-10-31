import torch
from torch import nn
import math

class PositionalEncoding(nn.Module):
    """
    Implementa la codifica posizionale sinusoidale.
    
    Poiché i Transformer elaborano i token in parallelo, non hanno una
    cognizione intrinseca dell'ordine della sequenza. Questa classe
    inietta informazioni sulla posizione (assoluta) di ciascun token
    sommando un vettore di codifica posizionale al suo embedding.
    
    La formula utilizza seni e coseni a diverse frequenze.
    PE(pos, 2i)   = sin(pos / 10000^(2i / dim))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / dim))
    
    Questo permette al modello di apprendere facilmente le relazioni di
    posizione relativa, poiché PE(pos+k) può essere rappresentato come
    una funzione lineare di PE(pos).
    """
    def __init__(self, dim: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Crea un tensore di forma (max_len, dim) per memorizzare i valori di PE
        pe = torch.zeros(max_len, dim)
        
        # Crea un tensore per le posizioni (0, 1, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calcola il termine divisore nelle formule
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        
        # Applica la formula: sin per le dimensioni pari, cos per le dispari
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Aggiunge una dimensione batch (1, max_len, dim)
        pe = pe.unsqueeze(0)
        
        # Registra 'pe' come buffer. I buffer sono parametri del modello
        # che non vengono aggiornati durante il backpropagation (come le medie
        # in BatchNorm), ma che devono essere salvati con lo stato del modello.
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x: Tensor, forma [batch_size, seq_len, embedding_dim]
        """
        # Somma la codifica posizionale (fino alla lunghezza della sequenza)
        # all'embedding di input.
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)