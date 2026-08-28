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
    # Understanding the Bias-Variance Tradeoff

    **Objective:** This notebook provides an interactive, hands-on demonstration of the bias-variance tradeoff, a fundamental concept in machine learning. You will explore how model complexity affects model error and how techniques like ensembling can change the balance between bias and variance.

    **Prerequisites:** You should be comfortable with basic machine learning concepts and have some familiarity with the scikit-learn library.

    *Inspired from https://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: The Theory

    In supervised learning, our goal is to build a model that learns a function `f(x)` from training data `(X, y)`. The model's prediction is denoted by `ŷ(x)`. The expected error of our model at a point `x` can be decomposed into three components:

    $$ \text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error (Noise)}^2 $$

    *(for a formal derivation of this, check out https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff#Derivation)*

    Let's break these down:

    *   **Bias:** This is the error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (underfitting). In simple terms, bias measures how far off the *average* prediction of our model is from the true value we are trying to predict.
        *   *A **high-bias** model is too simple and makes strong assumptions about the data.*

    *   **Variance:** This is the error from sensitivity to small fluctuations in the training set. High variance can cause an algorithm to model the random noise in the training data, rather than the intended outputs (overfitting). In simple terms, variance measures how much the prediction for a given point `x` varies if we were to retrain the model on different random subsets of our data.
        *   *A **high-variance** model is too complex and adapts too much to the training data.*

    *   **Irreducible Error (Noise):** This error is inherent to the data itself, as in most cases the generative process is actually of the form f(x) + $\epsilon$. It's the noise term that we can't fundamentally predict, no matter how good our model is.
    """)
    return


@app.cell
def _():
    # Core libraries
    import numpy as np
    import matplotlib.pyplot as plt

    # Scikit-learn for modeling
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import BaggingRegressor

    # PyTorch for the Neural Network part
    import torch
    import torch.nn as nn

    # Settings for the experiment
    n_repeat = 50  # Number of different random datasets to generate
    n_train = 50   # Size of each training set
    n_test = 1000  # Size of the test set
    noise = 0.1    # Standard deviation of the noise
    np.random.seed(42) # for reproducibility
    torch.manual_seed(42)
    return (
        BaggingRegressor,
        DecisionTreeRegressor,
        n_repeat,
        n_test,
        n_train,
        nn,
        noise,
        np,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2: The Experimental Setup

    To measure bias and variance, we need a controlled environment. We will:
    1.  Define a "ground truth" function, `f(x)`, which our models will try to learn.
    2.  Generate multiple noisy training sets from this function. This simulates the real-world scenario where we only have a random sample of data.
    3.  Create a single, large, noise-free test set to consistently evaluate the models.
    """)
    return


@app.cell
def _(n_repeat, n_test, n_train, noise, np, plt):
    # This is our "ground truth" function. It's non-linear and has some complexity.
    def f(x):
        x = x.ravel()
        return np.exp(-(x**2)) + 1.5 * np.exp(-((x - 2) ** 2))

    # This function generates many different datasets based on the ground truth function f.
    def generate_data(n_samples, noise, n_repeat=1):
        X = np.random.rand(n_samples) * 10 - 5
        X = np.sort(X)

        if n_repeat == 1:
            y = f(X) + np.random.normal(0.0, noise, n_samples)
        else:
            y = np.zeros((n_samples, n_repeat))

            for i in range(n_repeat):
                y[:, i] = f(X) + np.random.normal(0.0, noise, n_samples)

        X = X.reshape((n_samples, 1))

        return X, y

     # Generate the test sets, same X for all repeats, but different y
    X_test, y_test_sets = generate_data(n_test, noise, n_repeat)
    # the true function values for the test set
    y_test_true = f(X_test)

    # Let's visualize the true function and one random training set
    X_train_sample, y_train_sample = generate_data(n_train, noise)

    plt.figure(figsize=(8, 6))
    plt.plot(X_test, y_test_true, 'b-', label="True Function f(x)")
    plt.scatter(X_train_sample, y_train_sample, c='r', s=20, marker='o', label="Noisy Training Data Sample")
    plt.title("Ground Truth and a Sample Dataset")
    plt.legend()
    plt.show()
    return (
        X_test,
        X_train_sample,
        f,
        generate_data,
        y_test_sets,
        y_test_true,
        y_train_sample,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Simulation Framework

    The code cell below contains a function, `run_bias_variance_analysis`, that performs the entire experiment. You don't need to modify this code.

    **What it does:**
    1.  Takes a list of scikit-learn estimator models that you will define.
    2.  For each estimator:
        - It loops `n_repeat` times. In each loop, it generates a new noisy training set and fits the estimator to it.
        - It stores the predictions of each fitted estimator on the `X_test` set.
    3.  It then uses all the stored predictions to calculate the `bias²`, `variance`, and `total error`.
    4.  Finally, it generates the two key plots for you to analyze.
    """)
    return


@app.cell
def _(f, generate_data, n_repeat, n_test, n_train, noise, np, plt):
    def run_bias_variance_analysis(estimators, X_test, y_test_true, y_test_sets):
        """
        Performs the bias-variance decomposition experiment for a list of estimators.
        """
        n_estimators = len(estimators)
        plt.figure(figsize=(6 * n_estimators, 10))

        # Generate the different training sets: different X and y for each repeat
        X_train_sets, y_train_sets = [], []
        for i in range(n_repeat):
            X, y = generate_data(n_train, noise)
            X_train_sets.append(X)
            y_train_sets.append(y)
    

        # Loop over estimators to compare
        for n, (name, estimator) in enumerate(estimators):
            # Array to store predictions from each model trained on a different dataset
            y_predict = np.zeros((n_test, n_repeat))

            # Train n_repeat models and store their predictions
            for i in range(n_repeat):
                estimator.fit(X_train_sets[i], y_train_sets[i])
                y_predict[:, i] = estimator.predict(X_test)

            # calculate the error
            error = np.zeros(n_test)
            # we average the error over all datasets and all predictions
            for i in range(n_repeat):
                for j in range(n_repeat):
                    error += (y_test_sets[:, j] - y_predict[:, i]) ** 2

            error /= n_repeat * n_repeat

            # --- Bias-Variance Calculation ---
            # Bias^2 = (Average Prediction of y for each x value - True Value [we use f(X) to have the true response without noise])^2
            y_predict_mean = np.mean(y_predict, axis=1)
            bias_sq = (y_predict_mean - f(X_test)) ** 2
            # Variance = Average of (Prediction - Average Prediction)^2
            variance = np.var(y_predict, axis=1)
            # here we are getting the variance of the test sets, which is the noise variable defined above squared
            noise_sq = np.var(y_test_sets, axis=1).mean() 

            print(f"--- {name} ---")
            print(f"Average Error: {np.mean(error):.4f}")
            print(f"  Bias^2:      {np.mean(bias_sq):.4f}")
            print(f"  Variance:    {np.mean(variance):.4f}")
            print(f"  Noise^2:       {noise_sq:.4f}\n")

            # --- Plotting ---
            # Plot 1: Prediction "Beam"
            ax1 = plt.subplot(2, n_estimators, n + 1)
            ax1.plot(X_test, y_test_true, "b", label="$f(x)$ (True Function)")
            ax1.scatter(X_train_sets[0], y_train_sets[0], c='r', s=15, marker='.', label="Training Sample")
            # Plot the "beam" of all predictions
            for i in range(n_repeat):
                ax1.plot(X_test, y_predict[:, i], "g", alpha=0.1)
            ax1.plot(X_test, y_predict_mean, "c", lw=3, label=r"$\mathbb{E}[\^y(x)]$ (Avg. Prediction)")
            ax1.set_title(name)
            ax1.legend()

            # Plot 2: Error Decomposition
            ax2 = plt.subplot(2, n_estimators, n_estimators + n + 1)
            ax2.plot(X_test, error, "r", label="$Total Error(x)$")
            ax2.plot(X_test, bias_sq, "b", label="$bias^2(x)$")
            ax2.plot(X_test, variance, "g", label="$variance(x)$")
            ax2.plot(X_test, np.full(X_test.shape, noise_sq), "c", label="$noise^2$")
            ax2.set_title("Error Decomposition")
            ax2.legend()
    
        plt.tight_layout()
        plt.show()

    return (run_bias_variance_analysis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3: Core Demonstration with Decision Trees

    Now it's your turn. Your task is to define three different models using `DecisionTreeRegressor` and `BaggingRegressor`. The goal is to create models that exhibit:
    1.  High Bias
    2.  High Variance
    3.  Low Variance (through ensembling)
    """)
    return


@app.cell
def _(BaggingRegressor, DecisionTreeRegressor):
    ### EXERCISE 2: Define your models ###

    # Your task is to define a list of tuples, where each tuple is (name, model).
    # Follow the hints to create the three required models.

    # A very simple tree with a small max_depth will be rigid and have high bias.
    high_bias_model = DecisionTreeRegressor(max_depth=1)

    # A very complex tree with no depth limit will fit the noise and have high variance.
    high_variance_model = DecisionTreeRegressor(max_depth=10)

    # Bagging (Bootstrap Aggregating) averages many high-variance models to reduce variance.
    # We will use the high-variance tree as the base estimator.
    low_variance_model = BaggingRegressor(
        estimator=high_variance_model, n_estimators=10, random_state=42
    )

    # This list will be passed to our analysis function.
    estimators = [
        ("High Bias (Shallow Tree)", high_bias_model),
        ("High Variance (Deep Tree)", high_variance_model),
        ("Low Variance (Bagged Trees)", low_variance_model),
    ]
    return (estimators,)


@app.cell
def _(
    X_test,
    estimators,
    run_bias_variance_analysis,
    y_test_sets,
    y_test_true,
):
    # This cell runs the full analysis with the models you defined above.
    # You do not need to change this cell. Just run it after completing Exercise 2.
    run_bias_variance_analysis(estimators, X_test, y_test_true.ravel(), y_test_sets)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 3: Interpretation - Solution

    **1. Compare the `Bias²` and `Variance` for your "High Bias" and "High Variance" models. Did the results match what you expected? Explain why, referring to the `max_depth` parameter.**

    > The shallow tree has high bias because `max_depth=1` cannot represent the curved target function. Its predictions tend to be stable across training sets, so its variance is low. The deeper tree can fit much more of the target function and has lower bias, but its predictions change more when the noisy training sample changes, giving it higher variance.

    **2. Look at the "prediction beam" (the light green lines) for the "High Variance" model versus the "Bagged" model. Describe the difference you see.**

    > The deep-tree prediction beam is wider and more irregular: individual trees react strongly to their particular noisy training samples. The bagged prediction beam is narrower and smoother because each prediction averages multiple trees.

    **3. What was the effect of Bagging on the Bias? What was its effect on the Variance? What does this tell you about the primary strength of Bagging?**

    > Bagging primarily reduces variance. Its bias is usually similar to that of the individual base trees, although it can change slightly for a finite sample. Its strength is therefore making flexible, unstable learners more robust without imposing the strong restrictions of a shallow tree.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 4: Connection to Neural Networks

    The principles of bias and variance apply to *all* machine learning models, including neural networks.

    -   **High Bias in NNs:** Often caused by a network that is too simple (e.g., too few layers, too few neurons) or is under-trained (not enough epochs). It doesn't have the **capacity** to learn the true function.
    -   **High Variance in NNs:** Often caused by a network that is too complex (e.g., too many layers/neurons) or is over-trained. It has so much capacity that it starts to memorize the noise in the training data.

    Let's do a simpler, qualitative experiment. We will train two NNs on a single training set (for the sake of time) and observe their behavior. We won't be able to decompose the error as before but we can still gauge their performance against the true function f(X).
    """)
    return


@app.cell
def _(X_test, X_train_sample, nn, plt, torch, y_test_true, y_train_sample):
    # Helper function to train and plot a PyTorch NN model
    def train_and_plot_nn(model, name):
        """Trains a PyTorch model and plots its predictions."""
        # Convert data to torch tensors
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        X_train_torch = torch.from_numpy(X_train_sample).float().to(device)
        y_train_torch = torch.from_numpy(y_train_sample).float().reshape(-1, 1).to(device)
        X_test_torch = torch.from_numpy(X_test).float().to(device)

        # Simple training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        for epoch in range(2000):
            y_pred = model(X_train_torch)
            loss = loss_fn(y_pred, y_train_torch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # Plotting
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test_torch).cpu().numpy()

        plt.plot(X_test, y_test_true, 'b-', label="True Function f(x)")
        plt.scatter(X_train_sample, y_train_sample, c='r', s=20, marker='.', label="Noisy Training Data")
        plt.plot(X_test, y_pred_test, 'g-', lw=2, label="NN Prediction")
        plt.title(f"NN Model: {name}")
        plt.legend()
        plt.show()

    return (train_and_plot_nn,)


@app.cell
def _(nn):
    ### EXERCISE 4: Define your NN models ###

    # Your task is to define two simple neural networks using torch.nn.Sequential.
    # One should be low-complexity (high bias) and the other high-complexity (high variance).

    input_dim = 1
    output_dim = 1

    # A model with few neurons and layers lacks capacity.
    low_complexity_nn = nn.Sequential(
        nn.Linear(input_dim, 8),
        nn.ReLU(),
        nn.Linear(8, output_dim)
    )

    # A model with many neurons and layers has high capacity to memorize noise.
    high_complexity_nn = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, output_dim)
    )
    return high_complexity_nn, low_complexity_nn


@app.cell
def _(high_complexity_nn, low_complexity_nn, train_and_plot_nn):
    # Run the training and plotting for the low-complexity model
    train_and_plot_nn(low_complexity_nn, "Low Complexity (High Bias?)")

    # Run the training and plotting for the high-complexity model
    train_and_plot_nn(high_complexity_nn, "High Complexity (High Variance?)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Exercise 5: Final Interpretation - Solution

    **1. Which of your two neural networks fits the noisy red training data points more closely? What does this suggest about its variance?**

    The high-complexity network usually fits the noisy training points more closely. This flexibility makes its fitted function more sensitive to which noisy observations appeared in the training set, which is the hallmark of higher variance.

    **2. Which network provides a smoother, more generalized fit? What does this suggest about its bias?**

    The low-complexity network usually gives a smoother, more general fit. Its limited capacity prevents it from fitting all local variation, which corresponds to higher bias.

    **3. The high-complexity network shows "wiggles" in areas where there is no data. What real-world problem does this behavior represent, and why is it dangerous for a machine learning model?**

    The wiggles are overfitting: the model has learned noise or accidental details rather than the underlying relationship. This is dangerous because predictions can be poor and unstable for unseen inputs, especially in regions unlike the training data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion

    In this notebook, you have empirically verified the bias-variance tradeoff.

    **Key Takeaways:**
    *   **Simple models (like shallow trees or small NNs) are often high-bias and low-variance.** They are stable but may not be accurate because they can't capture the underlying complexity of the data.
    *   **Complex models (like deep trees or large NNs) are often low-bias and high-variance.** They are flexible enough to capture the true function but are also sensitive to noise, making them prone to overfitting.
    *   **Techniques like Bagging are powerful because they primarily reduce variance.** By averaging many different overfitted models, we can get a combined model that is both accurate and robust. This is a core idea behind modern ensemble methods like Random Forests.

    Finding the right model is always about finding the sweet spot in this tradeoff for your specific problem.
    """)
    return


if __name__ == "__main__":
    app.run()
