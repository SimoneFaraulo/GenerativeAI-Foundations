import torch
from torch import nn

class KVCache:
    """
    Gestisce il Key-Value Caching per l'inferenza autoregressiva.
    
    Durante la generazione token-by-token, non vogliamo ricalcolare le matrici
    Key e Value per i token passati, poiché rimangono costanti.
    Questa classe funge da buffer circolare o dinamico per memorizzare 
    K e V dei passaggi precedenti.
    """
    def __init__(self, max_seq_len: int, batch_size: int, num_heads: int, head_dim: int, device: torch.device):
        self.max_seq_len = max_seq_len
        # Inizializziamo i buffer per K e V con zeri (o vuoti).
        # Forma standard: (Batch, Num_Heads, Seq_Len, Head_Dim)
        self.k_cache = torch.zeros(batch_size, num_heads, max_seq_len, head_dim).to(device)
        self.v_cache = torch.zeros(batch_size, num_heads, max_seq_len, head_dim).to(device)
        
        # Puntatore alla posizione corrente nella sequenza
        self.current_seq_len = 0
        
    def update(self, new_k: torch.Tensor, new_v: torch.Tensor):
        """
        Aggiorna la cache con i nuovi K e V del token corrente e restituisce
        la sequenza completa di K e V concatenata.
        
        Input:
            new_k: (Batch, Num_Heads, 1, Head_Dim) - Key del nuovo token
            new_v: (Batch, Num_Heads, 1, Head_Dim) - Value del nuovo token
            
        Output:
            k_out, v_out: (Batch, Num_Heads, Current_Len, Head_Dim) - Tutta la storia
        """
        # Dimensione del nuovo chunk (solitamente 1 durante la generazione)
        seq_len_added = new_k.shape[2] 
        
        # Scriviamo i nuovi dati nella posizione corretta del buffer
        start_pos = self.current_seq_len
        end_pos = start_pos + seq_len_added
        
        self.k_cache[:, :, start_pos:end_pos, :] = new_k
        self.v_cache[:, :, start_pos:end_pos, :] = new_v
        
        self.current_seq_len = end_pos
        
        # Ritorniamo solo la parte valida della cache (fino al token corrente)
        # Questo verrà usato per calcolare l'attention su tutti i token passati
        return (self.k_cache[:, :, :self.current_seq_len, :], 
                self.v_cache[:, :, :self.current_seq_len, :])

    def reset(self):
        """Pulisce la cache per una nuova generazione."""
        self.current_seq_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()