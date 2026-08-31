# Machine Learning - Set 2: Hands-on VAE Concepts & Latent Space

**Objective:** To understand and compute key components of a Variational Autoencoder (VAE) using basic numerical operations. In this exercise, you will explore the reparameterization trick, calculate loss components (Reconstruction and KL Divergence), simulate latent variable collapse, and consider the role of the $\beta$-VAE.

**Tools:** You can use your preferred programming language (e.g., Python with NumPy/SciPy, MATLAB, Julia). You will need basic array/matrix operations, random number generation (standard normal), and math functions (exp, log, sum, mean). No dedicated ML frameworks are required.

**Setup:** We will work with a 2D input space and a 2D latent space ($d=2$). We will simulate the Encoder and Decoder using simple affine transformations (matrix multiplication + bias addition). Let $N=1000$ be the number of data points.

---

### **Notation**

*   $x \in \mathbb{R}^2$: Input data vector. `X_true` is the set of $N$ input vectors $\{x_1, ..., x_N \}$.
*   $\mu \in \mathbb{R}^2$: Mean vector of the latent distribution $q(z|x)$.
*   $\sigma \in \mathbb{R}^2$: Standard deviation vector of the latent distribution $q(z|x)$.
*   $\log \sigma^2 \in \mathbb{R}^2$: Log-variance vector (element-wise log of variance).
*   $z \in \mathbb{R}^2$: Latent space vector. `Z` is the set of $N$latent vectors $\{z_1, ..., z_N\}$.
*   $\epsilon \in \mathbb{R}^2$: Random noise vector sampled from the standard normal distribution $\mathcal{N}(0, I)$.
*   $x' \in \mathbb{R}^2$: Reconstructed data vector. `X_prime` is the set of $N$reconstructed vectors.
*   $\mathcal{N}(\mu, \Sigma)$: A multivariate Gaussian distribution with mean $\mu $and covariance $\Sigma$.
*   $\mathbf{W}_{\cdot}$: Weight matrices (all are 2x2).
*   $b_{\cdot}$: Bias vectors (all are 2D).
*   `@`: Matrix multiplication (e.g., `W @ x`).
*   `odot`: Element-wise (Hadamard) product.
*   $D_{KL}(q || p)$: The Kullback-Leibler (KL) divergence between distributions $q$ and $p$.

---

## 1. Data Generation

a. Generate $N=1000$ data points, forming the dataset `X_true`. Each point $x_i$ should be sampled from a 2D Gaussian distribution $\mathcal{N}(\mu_{data}, \Sigma_{data})$ with:
```math
\mu_{data} = \begin{pmatrix} 3.0 \\ 2.0 \end{pmatrix}, \quad \Sigma_{data} = \begin{pmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{pmatrix}
```
   Store these points in an $N \times 2$ array or matrix.

b. *Optional Visualization:* Create a scatter plot of your generated data points `$X_true$`.

## 2. Simulated VAE Parameters (Scenario 1: Standard)

We will simulate a VAE's Encoder and Decoder with the following fixed parameters.

*   **Encoder:** The encoder maps an input $x$ to the parameters of a latent distribution, with mean and log-variance:
   
```math
\mu = \mathbf{W}_{\mu} x + b_{\mu}
```
   
```math
    \log \sigma^2 = \mathbf{W}_{ls} x + b_{ls}
```
The "parameters" for this case are:
    
```math
\mathbf{W}_{\mu} = \begin{pmatrix} 0.5 & 0.0 \\ 0.0 & 0.5 \end{pmatrix}, \quad b_{\mu} = \begin{pmatrix} -0.5 \\ -0.25 \end{pmatrix}
```
```math
\mathbf{W}_{ls} = \begin{pmatrix} 0.1 & 0.0 \\ 0.0 & 0.1 \end{pmatrix}, \quad b_{ls} = \begin{pmatrix} -1.0 \\ -1.0 \end{pmatrix}
```

*   **Decoder:** The decoder maps a latent vector $z$ back to a reconstructed data point $x'$:
```math
x' = \mathbf{W}_{dec} z + b_{dec}
```

```math
\mathbf{W}_{dec} = \begin{pmatrix} 1.4 & 0.0 \\ 0.0 & 1.4 \end{pmatrix}, \quad b_{dec} = \begin{pmatrix} 1.5 \\ 1.0 \end{pmatrix}
```

> *Note: These parameters are fixed for this exercise. In a real VAE, these would be learned during training.*

## 3. VAE Forward Pass & Loss Calculation (Scenario 1)

Perform the following calculations for each data point $x_i$ in `X_true` using the parameters from Exercise 2.

a. **Encoding:** For each $x_i$, calculate the mean vector $\mu_i$ and the log-variance vector $\log \sigma^2_i$.

b. **Reparameterization Trick:**
   i.  Calculate the standard deviation vectors $\sigma_i$ using the formula:
       $\sigma_i = \exp(0.5 \odot \log \sigma^2_i)$

   ii. Sample $N=1000$ random noise vectors, $\{\epsilon_1, ..., \epsilon_N\}$, where each $\epsilon_i \sim \mathcal{N}(0, I)$.

   iii. Compute the set of latent vectors $Z = \{z_1, ..., z_N\}$ using the reparameterization formula: $z_i = \mu_i + \sigma_i \odot \epsilon_i$
   *Optional Visualization:* Create a scatter plot of the latent vectors $Z$.

c. **Decoding:** Calculate the reconstructed data points $X_{prime} = \{x'_1, ..., x'_N\}$ by passing the latent vectors $Z$ through the decoder.
   *Optional Visualization:* Create a scatter plot of the reconstructed points `X_prime` . Overlay it on the plot of `X_true` if possible.

d. **Loss Calculation:**
   i.  Calculate the **Reconstruction Loss**. We use the average Mean Squared Error (MSE):

```math
L_{recon} = \frac{1}{N} \sum_{i=1}^{N} ||x_i - x'_i||^2
```
      
where $||v||^2 = v_1^2 + v_2^2$.

   ii. Calculate the **KL Divergence** for each point with respect to the standard normal prior $\mathcal{N}(0, I)$. The formula for a diagonal Gaussian is:

```math
D_{KL,i} = \frac{1}{2} \sum_{j=1}^{2} (\sigma_{i,j}^2 + \mu_{i,j}^2 - \log(\sigma_{i,j}^2) - 1)
```

Remember that $\sigma_{i,j}^2 = \exp((\log \sigma^2_i)_j)$.
   iii. Calculate the average KL Divergence over all points:

```math
L_{KL} = \frac{1}{N} \sum_{i=1}^{N} D_{KL,i}
```
       
   iv. Calculate the total **VAE Loss** (this is the negative ELBO, with $\beta=1$):
```math
Loss = L_{recon} + L_{KL}
```

e. **Report your calculated values for $L_{recon}$, $L_{KL}$, and $Loss$.**

## 4. Simulating Latent Collapse (Scenario 2)

Now, we simulate a scenario where the encoder is "broken" and prioritizes matching the prior $\mathcal{N}(0, I)$ too strongly, ignoring the input data. We will use new encoder parameters, but keep the **original data** `X_true` and the **original decoder** parameters from Exercise 2.

*   **New Encoder Parameters ('Collapse'):**
```math
\mathbf{W}_{\mu, c} = \begin{pmatrix} 0.01 & 0.01 \\ -0.01 & -0.01 \end{pmatrix}, \quad b_{\mu, c} = \begin{pmatrix} 0.05 \\ -0.05 \end{pmatrix}
```
```math
\mathbf{W}_{ls, c} = \begin{pmatrix} 0.0 & 0.0 \\ 0.0 & 0.0 \end{pmatrix}, \quad b_{ls, c} = \begin{pmatrix} -0.1 \\ -0.1 \end{pmatrix}
```

a. **Repeat the full forward pass and loss calculation from Exercise 3 (steps a-d)**, but using these new 'collapse' encoder parameters instead of the original ones.

b. **Report the new values for $L_{recon, c}$, $L_{KL, c}$, and $Loss_c$.**

c. *Optional Visualization:* Create a scatter plot of the new latent vectors $Z_c$. How does it compare to the latent space plot from Scenario 1?

d. **Analysis:** Compare the loss components ($L_{recon}$ vs $L_{recon, c}$ and $L_{KL}$ vs $L_{KL, c}$) between Scenario 1 and Scenario 2. Describe what happened to the latent space ($Z_c$) and the reconstruction quality ($X'_{prime}$) in Scenario 2. Why is this phenomenon, known as "latent collapse," undesirable for a generative model?

