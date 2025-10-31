import torch
from torch import nn
from positional_encoding import PositionalEncoding
from transformer_encoder import Encoder
from transformer_decoder import Decoder
import math

class Transformer(nn.Module):
    """
    Modello Transformer Sequence-to-Sequence completo.
    
    Combina l'Encoder e il Decoder.
    
    Flusso dei dati:
    1. src (indici token) -> Embedding Sorgente -> + PositionalEncoding -> Encoder
    2. tgt (indici token) -> Embedding Target  -> + PositionalEncoding -> Decoder
    3. L'output dell'Encoder (K, V) viene passato al blocco Cross-Attention
       in ogni strato del Decoder.
    4. L'output del Decoder -> Layer Lineare -> Logits (probabilità sul vocabolario)
    """
    def __init__(self,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim: int = 512,
                 num_layers: int = 6,
                 heads: int = 8,
                 hidden_dim_multiplier: int = 4,
                 dropout: float = 0.1,
                 max_len: int = 5000):
        
        super().__init__()
        self.dim = dim
        
        # Embedding e Positional Encoding per la sorgente (Encoder)
        self.src_embedding = nn.Embedding(src_vocab_size, dim)
        self.src_pos_encoding = PositionalEncoding(dim, dropout, max_len)
        
        # Embedding e Positional Encoding per il target (Decoder)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, dim)
        self.tgt_pos_encoding = PositionalEncoding(dim, dropout, max_len)
        
        # Stack dell'Encoder
        self.encoder = Encoder(
            num_layers=num_layers,
            dim=dim,
            heads=heads,
            hidden_dim_multiplier=hidden_dim_multiplier,
            dropout=dropout
        )
        
        # Stack del Decoder
        self.decoder = Decoder(
            num_layers=num_layers,
            dim=dim,
            heads=heads,
            hidden_dim_multiplier=hidden_dim_multiplier,
            dropout=dropout
        )
        
        # Layer lineare finale per mappare l'output del decoder
        # alla dimensione del vocabolario target.
        self.output_linear = nn.Linear(dim, tgt_vocab_size)
        
        # Inizializzazione dei pesi (buona pratica)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def create_padding_mask(self, 
                            indices: torch.Tensor, 
                            pad_idx: int) -> torch.Tensor:
        """
        Crea una maschera per il padding.
        Input: (N, L) - indici dei token
        Output: (N, 1, L) - maschera (True dove c'è padding)
        """
        # Confronta gli indici con l'indice di padding
        mask = (indices == pad_idx) # (N, L)
        # Aggiunge una dimensione per il broadcasting con le teste (N, 1, L)
        return mask.unsqueeze(1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, src_pad_idx: int, tgt_pad_idx: int) -> torch.Tensor:
        """
        Input:
          src: (N, L_src) - Indici dei token della sequenza sorgente
          tgt: (N, L_tgt) - Indici dei token della sequenza target (es. shiftati a destra)
          src_pad_idx: Indice del token di padding per la sorgente
          tgt_pad_idx: Indice del token di padding per il target
        """
        
        # 1. Creare le maschere di padding
        # Maschera per l'encoder e per la cross-attention del decoder
        src_mask = self.create_padding_mask(src, src_pad_idx) # (N, 1, L_src)
        
        # Maschera per la self-attention del decoder
        # N.B. La maschera causale (triangolare) viene gestita
        # automaticamente dentro il DecoderLayer/MultiHeadAttention.
        # Questa maschera gestisce solo il padding del target.
        tgt_mask = self.create_padding_mask(tgt, tgt_pad_idx) # (N, 1, L_tgt)
        
        # 2. Processare la sorgente con l'Encoder
        
        # src_embed: (N, L_src) -> (N, L_src, D)
        src_embed = self.src_embedding(src) * math.sqrt(self.dim)
        src_embed = self.src_pos_encoding(src_embed)
        
        # encoder_output: (N, L_src, D)
        encoder_output = self.encoder(src_embed, src_mask)
        
        # 3. Processare il target con il Decoder
        
        # tgt_embed: (N, L_tgt) -> (N, L_tgt, D)
        tgt_embed = self.tgt_embedding(tgt) * math.sqrt(self.dim)
        tgt_embed = self.tgt_pos_encoding(tgt_embed)
        
        # decoder_output: (N, L_tgt, D)
        decoder_output = self.decoder(
            X=tgt_embed, 
            Z=encoder_output,   
            src_mask=src_mask,
            tgt_mask=tgt_mask # Usato per self-attention del decoder
        )
        
        # 4. Proiezione finale ai logit
        
        # logits: (N, L_tgt, D) -> (N, L_tgt, V_tgt)
        logits = self.output_linear(decoder_output)
        
        return logits

# --- Esempio di utilizzo ---
"""
Flusso End-to-End (Training)
Fase 1: Percorso ENCODER (Elaborazione Input Sorgente)
Testo Sorgente (stringa, es: "Il gatto") ->
    └-> [Tokenizer/Vocabolario Sorgente] -> Sequenza di Indici Sorgente (tensore di interi) ->
     -> [nn.Embedding Sorgente] -> Sequenza di Vettori (embeddings sorgente) -> [+ PositionalEncoding] ->
     -> Sequenza di Vettori Posizionati (input per l'encoder) 
     -> [Stack di N x EncoderLayer] (Ogni layer:)
            1. Self-Attention 
            2. Add&Norm 
            3. FFN 
            4. Add&Norm
     -> Output Encoder (MEMORIA) (Tensore [Batch, L_src, Dim]. Sarà K e V per il Decoder)

Fase 2: Percorso DECODER (Elaborazione Input Target)
Testo Target "Shifted-Right" (stringa, es: "<sos> The cat") 
    └-> [Tokenizer/Vocabolario Target] -> Sequenza di Indici Target (tensore di interi) ->
     -> [nn.Embedding Target] -> Sequenza di Vettori (embeddings target) -> [+ PositionalEncoding] ->
     -> Sequenza di Vettori Posizionati (input per il decoder) 
     -> [Stack di N x DecoderLayer] (Ogni layer riceve l'input precedente E la MEMORIA dell'Encoder ed esegue:)
            1. Masked Self-Attention (Q,K,V dai vettori target) 
            2. Add&Norm 
            3. Cross-Attention (Q dai vettori target, K,V dalla MEMORIA dell'Encoder) 
            4. Add&Norm 
            5. FFN
            6. Add&Norm
     -> Output Decoder (Tensore [Batch, L_tgt, Dim])

Fase 3: Percorso OUTPUT (Generazione Logits e Calcolo Loss)
     -> Output Decoder (Tensore [Batch, L_tgt, Dim]) -> [Layer Lineare Finale] (Proietta da Dim a Vocab_Size) ->
     -> Logits (Punteggi grezzi. Tensore [Batch, L_tgt, Vocab_Size]) ->
     -> [Funzione di Loss (es. CrossEntropyLoss)] (Confronta i Logits con gli Indici Target Reali, es: "The cat <eos>") 
     -> Valore di Loss (Scalare) (Un singolo numero usato per la backpropagation e l'aggiornamento dei pesi)
"""
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parametri del modello
    SRC_VOCAB_SIZE = 1000
    TGT_VOCAB_SIZE = 1200
    EMBED_DIM = 512
    NUM_LAYERS = 6
    HEADS = 8
    PAD_IDX = 0

    # Creazione del modello
    model = Transformer(
        src_vocab_size=SRC_VOCAB_SIZE,
        tgt_vocab_size=TGT_VOCAB_SIZE,
        dim=EMBED_DIM,
        num_layers=NUM_LAYERS,
        heads=HEADS,
        dropout=0.1
    ).to(device)

    # Creazione di dati di input fittizi
    N = 4 # Batch size
    L_SRC = 20 # Lunghezza sequenza sorgente
    L_TGT = 18 # Lunghezza sequenza target
    
    src = torch.randint(1, SRC_VOCAB_SIZE, (N, L_SRC)).to(device)
    tgt = torch.randint(1, TGT_VOCAB_SIZE, (N, L_TGT)).to(device)

    # Inserimento di padding fittizio
    src[0, 10:] = PAD_IDX
    tgt[1, 15:] = PAD_IDX
    
    print(f"Forma input sorgente (src): {src.shape}")
    print(f"Forma input target (tgt): {tgt.shape}")

    # Esecuzione del forward pass
    logits = model(src=src, tgt=tgt, src_pad_idx=PAD_IDX, tgt_pad_idx=PAD_IDX)
    
    print(f"Forma output logits: {logits.shape}")
    print("Esecuzione completata con successo!")