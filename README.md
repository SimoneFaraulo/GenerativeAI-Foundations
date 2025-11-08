# Generative AI Foundations

This repository contains a collection of PyTorch scripts that implement and explore the fundamental concepts of probability, information theory, and generative modeling from scratch. It serves as a practical "cookbook" for understanding the building blocks of modern generative AI.

## Key Concepts Explored

The project is organized into several key areas, each demonstrating a core principle:

### 1. Probability & Distributions (`/probability`)
- **1D & 2D Gaussians**: Visualizing and working with basic probability density functions (PDFs).
- **Conditional & Marginal Probability**: Numerically calculating `p(x|y)` and `p(x)` from a joint distribution.
- **Statistical Independence**: Visually demonstrating the difference between dependent and independent variables (`p(x,y) != p(x)p(y)`).

### 2. Information Theory (`/information_theory`)
- **Differential Entropy**: Numerically calculating the entropy of a continuous distribution.
- **Cross-Entropy & KL Divergence**: Measuring the "distance" between two distributions.

### 3. Sampling Methods (`/sampling`)
- **Inverse Transform Sampling**: An efficient method for generating samples from a known CDF.
- **Rejection Sampling**: A general algorithm for sampling from complex distributions.
- **Metropolis Algorithm**: An implementation of a fundamental Markov Chain Monte Carlo (MCMC) method for sampling from unnormalized densities.
- **Custom Distributions**: A custom `UniformDisk` distribution class is built to demonstrate extending PyTorch's capabilities.

### 4. Variable Transformations (`/var_transformation`)
- **Change of Variables Formula**: Implementing and visualizing how a PDF changes when the underlying random variable is transformed (`y = g(z)`).
- **Autograd for Jacobians**: Using PyTorch's `autograd` to explicitly calculate the Jacobian of a transformation.

### 5. Autoencoders (`/autoencoder`)
- **Deterministic Autoencoder**: A simple MLP-based autoencoder is trained to learn a compressed representation of data lying on a 2D manifold (a "sombrero" surface) embedded in 3D space.
- **Denoising Autoencoder**: An autoencoder is trained to reconstruct clean data from a noisy input, forcing it to learn more robust features of the underlying manifold.
- **Variational Autoencoder (VAE)**: Implementation of a VAE to learn a structured, probabilistic latent space. The scripts demonstrate:
  - **Balanced Training**: A well-behaved VAE with good reconstruction and generation.
  - **Posterior Collapse**: A failure mode where the model ignores the input, resulting in poor reconstructions.
  - **Poor Generation**: A failure mode where the model achieves excellent reconstruction but loses its ability to generate new, coherent data.
- **Conditional VAE (CVAE)**: Un'estensione del VAE che incorpora un'informazione di condizionamento (es. un'etichetta di classe). L'encoder apprende `q(z|x,c)` e il decoder apprende `p(x|z,c)`. Lo script dimostra come il modello può generare campioni appartenenti a una classe specifica a comando.
- **Vector Quantization VAE (VQ-VAE)**: Sostituisce lo spazio latente continuo con un "codebook" discreto. Dimostra la quantizzazione, l'addestramento tramite gradient pass-through e la commitment loss, e visualizza il fenomeno del "codebook collapse" in cui il modello utilizza solo un sottoinsieme degli archetipi disponibili.

### 6. Distribution Shift (`/distribution_shift`)
- **Importance Sampling**: A Monte Carlo method for estimating properties of a target distribution using samples from a different proposal distribution.
- **Density Ratio Estimation**: Using a classifier to learn the ratio `p(x)/q(x)` between two distributions.

### 7. Transformer Architecture (`/transformer`)
- **Positional Encoding**: Dimostrazione di come vengono iniettate le informazioni sulla posizione dei token, un passaggio fondamentale dato che il Transformer elabora i dati in parallelo.

- **Multi-Head Attention**: Implementazione del meccanismo di attenzione (sia self-attention che cross-attention) che costituisce il cuore dell'architettura Transformer.

- **Encoder & Decoder Layers**: Costruzione modulare dei singoli strati (Layer) dell'Encoder (con self-attention) e del Decoder (con masked self-attention e cross-attention).

- **Transformer Seq2Seq**: Assemblaggio dei componenti in un modello Encoder-Decoder completo, l'architettura fondamentale alla base dei moderni LLM e dei modelli di traduzione.

### 8. Generative Adversarial Network (GAN) (`/gan`)
- **Simple GAN**: Implementazione di una GAN di base (non condizionata) addestrata a generare punti in un piano 2D, imparando una distribuzione semplice.

- **Conditional GAN (CGAN)**: Estensione della GAN per includere informazioni di condizionamento (come etichette di classe).

- **Stabilizzazione dell'Addestramento**: Dimostrazione di tecniche comuni per stabilizzare l'addestramento avversariale, tra cui il **Label Smoothing** (per prevenire l'eccesso di confidenza del discriminatore) e l'**Instance Noise** (per "sfocare" le distribuzioni e prevenire gradienti nulli).
## How to Run

Each script is self-contained and can be run individually. Make sure you have PyTorch and Matplotlib installed:

```sh
pip install torch matplotlib scikit-learn numpy
```
Then, simply run any Python file from the terminal:
```sh
python practice/autoencoder/deterministic_ae.py
```