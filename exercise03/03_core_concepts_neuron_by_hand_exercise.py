import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Machine Learning Basics: A Single Neuron with NumPy

    **Objective.** Build and train a single sigmoid neuron from scratch. Work through the cells in order, replacing each `pass` with your answer. No dedicated ML framework is needed until the bonus section.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.datasets import make_moons

    return make_moons, mo, plt


@app.cell
def _(make_moons, plt):
    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
    y = y.reshape(-1, 1)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors="k")
    plt.title("Dataset C (Moons)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Understanding gradient descent

    ### 1a) Explain gradient descent in your own words
    Use the empty cell below for your answer.
    """)
    return


@app.cell
def _():
    # Write your explanation here.
    pass
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
def _():
    # Define cost_function(y_true, y_pred).
    pass
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1c) Derive the gradient

    We assume to have:
    - our predictions: $\hat y=\sigma(z)$,
    - inputs to the last layer: $z=X\vec{w}+b$,
    - the derivative $\sigma'(z)=\sigma(z)(1-\sigma(z))$

    Derive
    $\partial L/\partial w_j$ with the chain rule. Also derive the bias gradient. Use the cell below for your derivation.
    """)
    return


@app.cell
def _():
    # Write your derivation here (as a comment or mo.md(...)).
    pass
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Implementing a single neuron

    Complete each function below. Shapes: `X` is `(m, n_features)` and `weights` is `(n_features, 1)`.
    """)
    return


@app.cell
def _():
    # Define initialize_weights(input_dim). Use small random weights and a zero bias.
    pass
    return


@app.cell
def _():
    # Define sigmoid(x) = 1 / (1 + exp(-x)).
    pass
    return


@app.cell
def _():
    # Define forward_pass(X, weights, bias). Return z and sigmoid(z).
    pass
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Gradient descent

    With MSE and a sigmoid output, include both the factor from the MSE derivative and the sigmoid derivative when implementing the backward pass. Average the gradients over the batch.
    """)
    return


@app.cell
def _():
    # Define calculate_gradients(X, y_true, y_pred). Return dW and db.
    pass
    return


@app.cell
def _():
    # Define update_weights(weights, bias, dW, db, learning_rate).
    pass
    return


@app.cell
def _():
    # Define train_neuron(X, y, learning_rate, num_epochs).
    # Record one cost per epoch and print it every 100 epochs.
    pass
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Training and evaluation
    """)
    return


@app.cell
def _():
    # Choose hyperparameters, train the model, and assign:
    # trained_weights, trained_bias, cost_history
    pass
    return


@app.cell
def _():
    # Plot cost_history against epoch.
    pass
    return


@app.cell
def _():
    # Define evaluate_neuron(X, y, trained_weights, trained_bias).
    # Threshold predicted probabilities at 0.5 and return the accuracy.
    pass
    return


@app.cell
def _():
    # Evaluate and print the training accuracy.
    pass
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus: PyTorch

    Reimplement the neuron using `nn.Linear`, `nn.Sigmoid`, `nn.MSELoss`, and `torch.optim.SGD`. Compare its loss curve with the NumPy implementation. Use one or more cells below.
    """)
    return


@app.cell
def _():
    pass
    return


if __name__ == "__main__":
    app.run()
