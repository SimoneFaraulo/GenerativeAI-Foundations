import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    """
    Implementa un layer Linear con Low-Rank Adaptation (LoRA).
    
    In un layer standard: y = x @ W.T
    Con LoRA: y = x @ W.T + (x @ A.T @ B.T) * scaling
    
    Dove:
    - W è la matrice dei pesi pre-addestrata (congelata).
    - A è una matrice di proiezione verso il basso (rank r).
    - B è una matrice di proiezione verso l'alto (rank r).
    
    Vantaggio:
    W ha dimensione (d_in, d_out). Se d_in=1024, d_out=1024 -> ~1M parametri.
    Se usiamo rank r=8:
    A ha (1024, 8) = 8k parametri.
    B ha (8, 1024) = 8k parametri.
    Totale addestrabile = 16k (riduzione del 98%!).
    
    alpha/r: serve come fattore di scaling.
        1. Disaccoppia il Learning Rate dal Rank (r)
            Se aumenti r la matrice BA è il risultato di una somma di più prodotti scalari.
            Matematicamente, più è alto r, più la magnitudine dei valori in uscita da BA tende a crescere.
            Senza lo scaling, ogni volta che cambi r, dovresti cercare di nuovo il Learning Rate ottimale, perché i gradienti avrebbero scale diverse.
            Dividendo per r, si "normalizza" l'output in base alla dimensione del rango. 
            Moltiplicando per una costante alpha, si mantiene stabile l'energia del segnale.
            Risultato: Puoi cambiare r lasciando lo stesso Learning Rate, e il modello si addestrerà comunque bene.
        2. Agire come una "manopola del volume" 
            alpha è un iperparametro che decidi tu.
            Se alpha = r, lo scaling è 1.
            Se alpha < r, stai attenuando il contributo di LoRA.
            Se alpha > r, stai dando più peso ai nuovi pesi rispetto a quelli vecchi.
            Poiché all'inizio dell'addestramento LoRA viene inizializzato a 0 (perché la matrice B è inizializzata a zeri), 
            questo scaling aiuta a controllare la velocità con cui i nuovi pesi iniziano a influenzare l'output del modello rispetto ai pesi congelati W_0.
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, alpha: int = 16, dropout: float = 0.0):
        super().__init__()
        
        # 1. Il Layer Pre-addestrato (Simulato)
        # In un caso reale, questo verrebbe caricato da un modello esistente e congelato.
        self.pretrained = nn.Linear(in_features, out_features)
        # Congeliamo i pesi (non verranno aggiornati dalla backprop)
        self.pretrained.weight.requires_grad = False
        if self.pretrained.bias is not None:
            self.pretrained.bias.requires_grad = False
            
        # 2. Le Matrici LoRA (Addestrabili)
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # Matrice A: Proiezione verso il basso (in -> r)
        # Inizializzazione Gaussiana (Kaiming/Normal)
        self.lora_A = nn.Parameter(torch.randn(r, in_features))
        
        # Matrice B: Proiezione verso l'alto (r -> out)
        # Inizializzata a Zeri. Perché?
        # All'inizio del training, vogliamo che il contributo di LoRA sia zero
        # in modo che l'output sia identico a quello del modello pre-addestrato.
        # y = Wx + 0. Se B fosse random, inizieremmo con del rumore aggiunto.
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Reset dei parametri A (B è già a zero)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: output pre-addestrato + output LoRA
        """
        # 1. Calcolo del percorso "congelato"
        pretrained_out = self.pretrained(x)
        
        # 2. Calcolo del percorso LoRA (Low-Rank)
        # x -> dropout -> A -> B -> scaling
        lora_out = (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        
        # 3. Somma finale
        return pretrained_out + lora_out

    def merge_weights(self):
        """
        Opzionale: Unisce i pesi LoRA nel layer originale per l'inferenza.
        W_new = W_old + (B @ A) * scaling
        Questo rimuove l'overhead computazionale durante l'inferenza!
        """
        if self.r > 0:
            # Calcoliamo il delta W
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            # Aggiorniamo i pesi originali
            self.pretrained.weight.data += delta_w
            print("Pesi LoRA uniti nel layer principale.")