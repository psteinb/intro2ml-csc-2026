import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Building Autoencoders with PyTorch

    **Objective:** In this notebook, you will build and train two types of autoencoders—a standard autoencoder and a variational autoencoder (VAE)—to perform dimensionality reduction and generation on the MNIST dataset.

    **Content Credit:** This notebook is a guided exercise adapted from the excellent blog post ["Intuitively Understanding Variational Autoencoders"](https://avandekleut.github.io/vae/) by **Alexander Van de Kleut**. We highly recommend reading the original post for a deeper dive.
    """)
    return


@app.cell
def _():
    # Standard imports
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils
    import torch.distributions
    import torchvision
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams['figure.dpi'] = 200
    # Plotting settings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set device
    print(f'Using device: {device}')
    return device, nn, np, plt, torch, torchvision


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: The "Why" - Motivation and Dimensionality Reduction

    Imagine we have a large, high-dimensional dataset, like the MNIST dataset of handwritten digits. Each image is 28x28 pixels, meaning each data point has 784 dimensions.

    The **manifold hypothesis** suggests that real-world high-dimensional data often lies on a much lower-dimensional "manifold" embedded within that high-dimensional space. This means that while the data has 784 dimensions, its underlying structure might be described by just a few.

    This is the motivation for **dimensionality reduction**: taking high-dimensional data and projecting it onto a lower-dimensional surface. This is a form of **unsupervised learning**, where the goal is not to predict a label, but to learn the underlying structure of the data itself.

    **Autoencoders** are a type of neural network designed specifically for this task.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2: Standard Autoencoders

    An autoencoder is composed of two networks: an **encoder** and a **decoder**.

    1.  **Encoder ($e$):** This network learns a transformation $e: \mathcal{X} \rightarrow \mathcal{Z}$ that projects the high-dimensional input data from space $\mathcal{X}$ (e.g., 784 dimensions) to a lower-dimensional **latent space** $\mathcal{Z}$ (e.g., 2 dimensions). The output, $z = e(x)$, is called a **latent vector**.

    2.  **Decoder ($d$):** This network learns a transformation $d: \mathcal{Z} \rightarrow \mathcal{X}$ that projects the latent vector back into the original high-dimensional space, attempting to reconstruct the original input: $\hat{x} = d(z) = d(e(x))$.

    The entire autoencoder is trained end-to-end by minimizing a **reconstruction loss**, which measures the difference between the original input $x$ and the reconstructed output $\hat{x}$.

    ![autoencoder](autoencoder-architecture.png)
    from https://lilianweng.github.io/posts/2018-08-12-vae/
    """)
    return


@app.cell
def _(nn):
    ### EXERCISE 1: Implement the Encoder ###

    # The Encoder takes a 784-dimensional image and maps it to a `latent_dims`-dimensional vector.
    # Your task is to define the two linear layers.

    class Encoder(nn.Module):
        def __init__(self, latent_dims):
            super(Encoder, self).__init__()
            # YOUR CODE HERE: Define a NN with input 784 to 512 neurons, then from 512 to the latent dimension. The activation for the first layer should be relu, while the second one should not be activated

        def forward(self, x):
            # The forward pass defines how data flows through the layers.
            # First, remmber to flatten the 28x28 image into a 784-dim vector.
            # TO DO
            return x

    return (Encoder,)


@app.cell
def _(nn):
    ### EXERCISE 2: Implement the Decoder ###

    # The Decoder takes a latent vector and attempts to reconstruct the original 784-dimensional image.
    # Your task is to define the two linear layers.

    class Decoder(nn.Module):
        def __init__(self, latent_dims):
            super(Decoder, self).__init__()
            # YOUR CODE HERE: Define a NN from the latent dimension to 512 neurons, then from 512 to 784
            # Activate the first layer with relu and the output with Sigmoid (to get pixel intensity between 0 and 1)

        def forward(self, z):
            # TO DO
            # Finally, we reshape the 784-dim vector back to a 28x28 image.
            return z.reshape((-1, 1, 28, 28))

    return (Decoder,)


@app.cell
def _(Decoder, Encoder, nn):
    # We put everything together
    class Autoencoder(nn.Module):
        def __init__(self, latent_dims):
            super(Autoencoder, self).__init__()
            self.encoder = Encoder(latent_dims)
            self.decoder = Decoder(latent_dims)

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    return (Autoencoder,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3: Training and Evaluating the Autoencoder

    Now we'll write a function to train the autoencoder. The training process involves:
    1.  Passing a batch of images through the autoencoder to get reconstructions.
    2.  Calculating the reconstruction loss (we'll use Mean Squared Error).
    3.  Backpropagating the loss and updating the model's weights.
    """)
    return


@app.cell
def _(device, torch):
    ### EXERCISE 4: Implement the Training Loss ###

    def train(autoencoder, data, epochs=20):
        opt = torch.optim.Adam(autoencoder.parameters())
        for epoch in range(epochs):
            for x, y in data:
                x = x.to(device)
                opt.zero_grad()
            
                # Get the model's reconstruction
                x_hat = ...
            
                # YOUR CODE HERE: Calculate the reconstruction loss.
                # This should be the Mean Squared Error (MSE) between the
                # original input `x` and the reconstruction `x_hat`.
                # sum over the batch examples
                loss = ...
            
                loss.backward()
                opt.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        return autoencoder

    return (train,)


@app.cell
def _(Autoencoder, device, torch, torchvision, train):
    # --- Setup and Run Training ---

    # Set the latent dimension to 2 so we can visualize it.
    latent_dims = 2
    autoencoder = Autoencoder(latent_dims).to(device)

    # Load the MNIST dataset
    data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('./data',
                   transform=torchvision.transforms.ToTensor(),
                   download=True),
            batch_size=128,
            shuffle=True)

    # Train the autoencoder
    autoencoder = train(autoencoder, data, epochs=5) # Using 5 epochs for speed
    return autoencoder, data, latent_dims


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Evaluating the Trained Autoencoder

    Once trained, we can do two interesting things:
    1.  **Visualize the Latent Space:** We can encode a batch of images and create a scatter plot of the resulting 2D latent vectors. We'll color-code the points by their digit label to see if the model has learned to cluster similar digits.
    2.  **Generate from the Latent Space:** We can sample points from the 2D latent space and pass them through the decoder. This allows us to see what kind of images the decoder generates from different regions of the space.
    """)
    return


@app.cell
def _(device, np, plt, torch):
    # We provide these helper functions for you.
    def plot_latent(autoencoder, data, num_batches=100):
        plt.figure(figsize=(8, 6))
        for i, (x, y) in enumerate(data):
            z = autoencoder.encoder(x.to(device))
            z = z.to('cpu').detach().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10', s=10)
            if i >= num_batches:
                break
        plt.colorbar()
        plt.title("Latent Space of the Standard Autoencoder")
        plt.show()

    def plot_reconstructed(autoencoder, r0=(-5, 10), r1=(-10, 5), n=12):
        w = 28
        img = np.zeros((n*w, n*w))
        plt.figure(figsize=(8, 8))
        for i, y in enumerate(np.linspace(*r1, n)):
            for j, x in enumerate(np.linspace(*r0, n)):
                z = torch.Tensor([[x, y]]).to(device)
                x_hat = autoencoder.decoder(z)
                x_hat = x_hat.reshape(28, 28).to('cpu').detach().numpy()
                img[(n-1-i)*w:(n-1-i+1)*w, j*w:(j+1)*w] = x_hat
        plt.imshow(img, extent=[*r0, *r1], cmap='gray')
        plt.title("Reconstructions from Latent Space")
        plt.show()

    return plot_latent, plot_reconstructed


@app.cell
def _(autoencoder, data, plot_latent, plot_reconstructed):
    # Plot the latent space
    plot_latent(autoencoder, data)

    # Plot reconstructions from the latent space
    plot_reconstructed(autoencoder)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analysis: The Problem with Standard Autoencoders

    Look at the plots above. The latent space shows clear clusters for different digits, which is great!

    However, look at the "Reconstructions" plot. You'll notice that there are "gaps" between the clusters in the latent space. If we sample a latent vector from one of these gaps (e.g., the top-left corner), the decoder produces a meaningless, blurry image. This is because the decoder was never trained on latent vectors from these regions.

    This makes standard autoencoders poor **generative models**. We can't just sample a random latent vector and expect a realistic output.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 4: Variational Autoencoders (VAEs)

    **Variational Autoencoders (VAEs)** solve this problem by making the latent space more continuous and structured.

    Instead of mapping an input `x` to a single latent point `z`, the VAE encoder maps it to a **probability distribution**—specifically, a Gaussian distribution defined by a mean `μ` and a standard deviation `σ`. A latent vector `z` is then *sampled* from this distribution.

    ![vae](vae-gaussian.png)
    from https://lilianweng.github.io/posts/2018-08-12-vae/

    This has two effects:
    1.  **Robust Decoder:** The decoder learns to reconstruct images from a wider range of latent vectors, not just single points, making it more robust.
    2.  **Regularized Latent Space:** We add a new term to the loss function: the **Kullback-Leibler (KL) Divergence**. This term penalizes the learned distributions (`μ`, `σ`) for being too far from a standard normal distribution ($\mathcal{N}(0, 1)$). This forces all the encoded distributions to cluster around the center of the latent space, filling in the gaps and creating a smooth, continuous space. It is given by
    $$\mathbb{KL}\left( \mathcal{N}(\mu, \sigma) \parallel \mathcal{N}(0, 1) \right) = \sum_{x \in X} \left( \sigma^2 + \mu^2 - \log \sigma - \frac{1}{2} \right)$$

    The **total VAE loss** is: `Reconstruction Loss + KL Divergence Loss`
    """)
    return


@app.cell
def _(Decoder, device, nn, torch):
    ### EXERCISE 5: Implement the Variational Encoder ###

    # The Decoder class remains the same. We only need to change the Encoder.

    class VariationalEncoder(nn.Module):
        def __init__(self, latent_dims):
            super(VariationalEncoder, self).__init__()
  
            # YOUR CODE HERE: The VAE encoder needs one input layer of dim 784, relu activated, and two output layers, not activated: 
            # one for the mean (mu) and one for the log of the variance (log_sigma).
            # Both should map from 512 to latent_dims.

            self.N = torch.distributions.Normal(0, 1)
            self.N.loc = self.N.loc.to(device) # for sampling on GPU
            self.N.scale = self.N.scale.to(device)
            self.kl = 0

        def forward(self, x):
            # remember to flatten the inputs
        
        
            # Get mu and sigma from the final layers
            mu =  ...
            # use exp on log_sigma to ensure sigma is always positive
            sigma = ...
        
            # YOUR CODE HERE: Sample the latent vector `z` from the distribution N(mu, sigma).
            z = ...
        
            # YOUR CODE HERE: Calculate the KL divergence.
            self.kl = ...
        
            return z

    class VariationalAutoencoder(nn.Module):
        def __init__(self, latent_dims):
            super(VariationalAutoencoder, self).__init__()
            self.encoder = VariationalEncoder(latent_dims)
            self.decoder = Decoder(latent_dims) # The same decoder as before!

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    return (VariationalAutoencoder,)


@app.cell
def _(device, torch):
    ### EXERCISE 6: Implement the VAE Training Loss ###

    def train_vae(vae, data, epochs=20):
        opt = torch.optim.Adam(vae.parameters())
        for epoch in range(epochs):
            for x, y in data:
                x = x.to(device)
                opt.zero_grad()
            
                # preidct using the model
                x_hat = ...
            
                # YOUR CODE HERE: Calculate the total VAE loss.
                # It's the reconstruction loss (like before) PLUS the KL divergence,
                # which we stored as an attribute in our VariationalEncoder.
                recon_loss = ...
                kl_loss = ...
                loss = recon_loss + kl_loss
            
                loss.backward()
                opt.step()
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, KL Div: {vae.encoder.kl.item():.4f}")
        return vae

    return (train_vae,)


@app.cell
def _(
    VariationalAutoencoder,
    data,
    device,
    latent_dims,
    plot_latent,
    plot_reconstructed,
    train_vae,
):
    # --- Setup and Run VAE Training ---
    vae = VariationalAutoencoder(latent_dims).to(device)
    vae = train_vae(vae, data, epochs=5) # Using 5 epochs for speed

    # --- Plot the VAE results ---
    print("\n--- VAE Results ---")
    plot_latent(vae, data)
    plot_reconstructed(vae, r0=(-2, 2), r1=(-2, 2))
    return (vae,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Analysis: The VAE Improvement

    Compare the VAE plots to the standard AE plots. What do you notice?

    1.  **The Latent Space is Compact:** The latent vectors are now tightly packed in a spherical cloud around the origin (0,0). The KL divergence forced this to happen.
    2.  **The Reconstructions are Smooth:** The space between digits is no longer a meaningless void. Instead, you see smooth transitions from one digit to another. The decoder has learned to generate plausible (if blurry) images from any point in this central region. The VAE is a true generative model!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 5: Fun with Interpolation

    Because the VAE's latent space is smooth, we can do fun things like interpolate between two digits. We'll take a '1' and a '0', find their latent vectors `z1` and `z2`, and then decode points along the straight line connecting `z1` and `z2`.
    """)
    return


@app.cell
def _(autoencoder, data, device, np, plt, torch, vae):
    ### EXERCISE 7: Implement Latent Space Interpolation ###

    def interpolate(autoencoder, x_1, x_2, n=12):
        # Get the latent vectors for the two input images
        z_1 = autoencoder.encoder(x_1)
        z_2 = autoencoder.encoder(x_2)
    
        # YOUR CODE HERE: Create a list of `n` interpolated latent vectors.
        # Start at z_1 and linearly move to z_2.
        # Hint: z = z_1 + t * (z_2 - z_1), where t goes from 0 to 1.
        z = torch.stack([z_1 + (z_2 - z_1)*t for t in np.linspace(0, 1, n)])
    
        # Decode the interpolated latent vectors
        interpolate_list = autoencoder.decoder(z)
        interpolate_list = interpolate_list.to('cpu').detach().numpy()

        # Plot the results
        w = 28
        img = np.zeros((w, n*w))
        for i, x_hat in enumerate(interpolate_list):
            img[:, i*w:(i+1)*w] = x_hat.reshape(28, 28)
        plt.figure(figsize=(10, 1))
        plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()

    # Grab a batch of data to find a '1' and a '0'
    x, y = next(iter(data))
    x_1 = x[y == 1][0].to(device).unsqueeze(0) # Find first '1'
    x_2 = x[y == 0][0].to(device).unsqueeze(0) # Find first '0'

    print("Interpolating with the VAE (smooth):")
    interpolate(vae, x_1, x_2, n=20)

    print("\nInterpolating with the standard AE (gappy):")
    interpolate(autoencoder, x_1, x_2, n=20)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    In this notebook, you have successfully built, trained, and evaluated both a standard autoencoder and a variational autoencoder.

    **Key Takeaways:**
    *   **Standard Autoencoders** are effective for dimensionality reduction and learning compressed representations, but their latent spaces can be disjoint and irregular.
    *   **Variational Autoencoders** use a probabilistic encoder and a KL divergence loss to enforce a regular, continuous structure on the latent space.
    *   This regular structure makes **VAEs powerful generative models**, capable of creating new, realistic data by sampling from the latent space and decoding. The smooth transitions you saw during interpolation are a direct result of this improved structure.
    """)
    return


if __name__ == "__main__":
    app.run()

