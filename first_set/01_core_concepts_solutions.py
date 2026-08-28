import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.datasets import make_moons

    return make_moons, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Machine Learning Basics: A Single Neuron with NumPy

    **Objective.** Build and train a single sigmoid neuron from scratch. Work through the cells in order, replacing each `pass` with your answer. No dedicated ML framework is needed until the bonus section.
    """)
    return


@app.cell
def _(make_moons, plt):
    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
    y = y.reshape(-1, 1)

    print(f"input data x {X.shape} and targets y {y.shape}")

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="k")
    plt.title("Dataset C (Moons)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Understanding gradient descent

    ### 1a) Explain gradient descent in your own words

    Gradient descent repeatedly evaluates how the loss changes as each parameter changes, then moves the parameters a small distance in the *opposite* direction. Repeating this process aims to reach a set of parameters with low loss.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1b) Mean squared error

    Implement
    $$L(W,b) = \frac{1}{m}\sum_{i=1}^{m}(y^{(i)}-\hat y^{(i)})^2.$$
    """)
    return


@app.cell
def _(np):
    def cost_function(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    return (cost_function,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1c) Derive the gradient

    We use mean squared error (MSE) with $i$ being the subscript for a single sample in the dataset:
    $$L = \frac{1}{m}\sum_i(y_i-\hat y_i)^2 = \frac{1}{m}\sum_i\ell_i.$$

    We assume to have for each sample $i$ in the dataset $X$:
    - the loss function: $\ell_i = f(y_i,\hat{y}_i) = (y_i-\hat y_i)^2$
    - the predictions: $\hat y_i=\sigma(z_i)$,
    - inputs to the last layer: $z=X\vec{w}+b$ or per sample $z_i=\vec{x_i}\cdot\vec{w}+b$,
    - the derivative of the sigmoid allows a neat trick: $\sigma'(z_i)=\sigma(z_i)(1-\sigma(z_i))$

    Derive
    $\partial L/\partial w_j$ with the chain rule for each weight $w_j$ in $\vec{w}$. Also derive the bias gradient $\partial L/\partial b$. Use the cell below for your derivation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The bias term is easier as it is a global parameter to optimize here:

    $$\frac{\partial \ell_i}{\partial b} = \frac{\partial \ell_i}{\partial \hat{y}_i}\cdot\frac{\partial \hat{y}_i}{\partial z_i}\cdot\frac{\partial z_i}{\partial b} = -2(y_i - \hat{y}_i)\cdot \sigma(z_i)(1-\sigma(z_i)) \cdot 1 = -2(y_i - \hat{y}_i)\cdot\hat{y}_i(1-\hat{y}_i)\cdot 1$$

    The derivative for each weight entry $w_j$ is a bit more tricky:

    $$\frac{\partial \ell_i}{\partial w_j} = \frac{\partial \ell_i}{\partial \hat{y}_i}\cdot\frac{\partial \hat{y}_i}{\partial z_i}\cdot\frac{\partial z_i}{\partial w_j} = -2(y_i - \hat{y}_i)\cdot\hat{y}_i(1-\hat{y}_i) \cdot (x_{i,j})$$

    Averaging over the batch (avergaging over all indices $i$) gives the weight gradient which will later be used in the weight update rule:

    $$ \frac{\partial L}{\partial w_j} = \frac{\partial \frac{1}{m}\sum_i\ell_i}{\partial w_j} = \frac{1}{m}\sum_i\frac{\partial \ell_i}{\partial w_j}$$

    The same summation holds for $\frac{\partial L}{\partial b}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Implementing a single neuron

    Complete each function below. Shapes: `X` is `(m, n_features)` and `weights` is `(n_features, 1)`.
    """)
    return


@app.cell
def _(np):
    def initialize_weights(input_dim):
        weights = 0.1 * np.random.randn(input_dim, 1)
        bias = 0.0
        return weights, bias

    return (initialize_weights,)


@app.cell
def _(np):
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    return (sigmoid,)


@app.cell
def _(sigmoid):
    def forward_pass(X, weights, bias):
        z = X @ weights + bias #X @ weights refers to the matrix-vector product of X and the weights
        return z, sigmoid(z)

    return (forward_pass,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Gradient descent

    With MSE and a sigmoid output, include both the factor from the MSE derivative and the sigmoid derivative when implementing the backward pass. Average the gradients over the batch.
    """)
    return


@app.cell
def _(np):
    def calculate_gradients(X, y_true, y_pred):
        # dL/dz includes the MSE and sigmoid derivatives for every sample.
    
        # we can reuse dz as it occurs in both dW and db
        # NB. y_true and y_pred are both vectors
        # here: dz is equivalent to dl_i/dz_i
        dz = -2 * (y_true - y_pred) * y_pred * (1 - y_pred)

        # the size of our dataset, i.e. the number of samples in it
        m = len(y_true)

        # we implicitely do the sum over all x_i to compute the gradient 
        # for w_j as contained in W = (w_0, w_1).T
        dW = (1/m)*(X.T @ dz)

        # we reuse dz from above. As dz_i/db = 1, we can directly execute the summation.
        db = float(np.mean(dz))
        return dW, db

    return (calculate_gradients,)


@app.function
def update_weights(weights, bias, dW, db, learning_rate):
    new_weights = weights - learning_rate * dW
    new_bias = bias - learning_rate * db
    return new_weights, new_bias


@app.cell
def _(calculate_gradients, cost_function, forward_pass, initialize_weights):
    def train_neuron(X, y, learning_rate, num_epochs):
        weights, bias = initialize_weights(X.shape[1])
        cost_history = []
        for epoch in range(num_epochs):
            _, y_pred = forward_pass(X, weights, bias)
            cost = cost_function(y, y_pred)
            cost_history.append(cost)
            dW, db = calculate_gradients(X, y, y_pred)
            weights, bias = update_weights(weights, bias, dW, db, learning_rate)
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")
        return weights, bias, cost_history

    return (train_neuron,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Training and evaluation
    """)
    return


@app.cell
def _(X, train_neuron, y):
    learning_rate = 0.1
    num_epochs = 1_000
    trained_weights, trained_bias, cost_history = train_neuron(X, y, learning_rate, num_epochs)
    print(f"Final weights:\n{trained_weights}")
    print(f"Final bias: {trained_bias:.4f}")
    return cost_history, learning_rate, trained_bias, trained_weights


@app.cell
def _(cost_history, plt):
    plt.figure(figsize=(8, 6))
    plt.plot(cost_history)
    plt.title("Cost Function Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(forward_pass, np):
    def evaluate_neuron(X, y, trained_weights, trained_bias):
        _, y_pred_proba = forward_pass(X, trained_weights, trained_bias)
        y_pred_class = (y_pred_proba >= 0.5).astype(int)
        return float(np.mean(y_pred_class == y))

    return (evaluate_neuron,)


@app.cell
def _(X, evaluate_neuron, trained_bias, trained_weights, y):
    accuracy = evaluate_neuron(X, y, trained_weights, trained_bias)
    print(f"Final training accuracy: {accuracy * 100:.2f}%")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus: PyTorch

    PyTorch supplies the same pieces as pre-built components: `nn.Linear` is the linear step, `nn.Sigmoid` is the activation, `nn.MSELoss` is the cost, and `loss.backward()` calculates gradients automatically.
    """)
    return


@app.cell
def _(X, cost_history, learning_rate, plt, y):
    import torch
    import torch.nn as nn

    X_torch = torch.from_numpy(X).float()
    y_torch = torch.from_numpy(y).float()
    model = nn.Sequential(nn.Linear(X.shape[1], 1), nn.Sigmoid())
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    torch_cost_history = []

    for _epoch in range(1_000):
        outputs = model(X_torch)
        loss = criterion(outputs, y_torch)
        torch_cost_history.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    plt.figure(figsize=(10, 4))
    plt.plot(cost_history, label="NumPy")
    plt.plot(torch_cost_history, label="PyTorch")
    plt.title("MSE Cost Histories")
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.legend()
    plt.grid(True)
    plt.show()

    torch_weights = model[0].weight.detach().numpy().T
    torch_bias = float(model[0].bias.detach().numpy()[0])
    return torch_bias, torch_weights


@app.cell
def _(X, evaluate_neuron, torch_bias, torch_weights, y):
    accuracy_torch = evaluate_neuron(X, y, torch_weights, torch_bias)
    print(f"Final PyTorch training accuracy: {accuracy_torch * 100:.2f}%")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
