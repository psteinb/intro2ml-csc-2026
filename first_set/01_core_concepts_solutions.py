import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_moons

    return make_moons, np, plt


@app.cell
def _(make_moons, plt):
    # Generate the dataset
    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

    # Reshape y to be a column vector (m, 1) instead of a row vector (m,)
    # This is a common practice that simplifies matrix operations later on.
    y = y.reshape(-1, 1)

    # Visualize the dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.title('Dataset C (Moons)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    return X, y


@app.cell
def _(np):
    def cost_function(y_true, y_pred):
        """
        Calculates the Mean Squared Error cost.
    
        Args:
            y_true (np.array): Array of true labels, shape (m, 1).
            y_pred (np.array): Array of predicted labels, shape (m, 1).
        
        Returns:
            float: The mean squared error.
        """
        # Your implementation here
        # Hint: m is the number of samples, which is the length of y_true.
        m = len(y_true)
        cost = np.sum((y_true - y_pred)**2)/m
    
        return cost

    return (cost_function,)


@app.cell
def _(np):
    def initialize_weights(input_dim):
        """
        Initializes weights with small random numbers and bias with zero.
    
        Args:
            input_dim (int): The number of features in the input data (e.g., 2 for our dataset).
        
        Returns:
            tuple: A tuple containing:
                - weights (np.array): A weight vector of shape (input_dim, 1).
                - bias (float): A scalar bias term, initialized to 0.
        """
        # Your implementation here

        weights = 0.1*np.random.randn(input_dim,1)
        bias = 0.0 # Replace
    
        return weights, bias

    return (initialize_weights,)


@app.cell
def _(np):
    def sigmoid(x):
        """
        Computes the sigmoid function.
    
        Args:
            x (np.array or float): Input value(s).
        
        Returns:
            np.array or float: The sigmoid of x.
        """
        out = 1/(1 + np.exp(-x))
        return out

    return (sigmoid,)


@app.cell
def _(np, sigmoid):
    def forward_pass(X, weights, bias):
        """
        Computes the forward pass of the neuron.
    
        Args:
            X (np.array): Input data, shape (m, n_features).
            weights (np.array): Weight vector, shape (n_features, 1).
            bias (float): Bias term.
        
        Returns:
            tuple: A tuple containing:
                - z (np.array): The linear combination, shape (m, 1).
                - y_pred (np.array): The prediction (activation), shape (m, 1).
        """
        # Your implementation here
        # Hint: Use the correct operation (scalar product) for multiplying X and weights
        z = np.dot(X, weights) + bias
        y_pred = sigmoid(z)
    
        return z, y_pred

    return (forward_pass,)


@app.cell
def _(np):
    def calculate_gradients(X, y_true, y_pred):
        """
        Calculates the gradients of the cost function w.r.t. weights and bias.
    
        Args:
            X (np.array): Input data, shape (m, n_features).
            y_true (np.array): True labels, shape (m, 1).
            y_pred (np.array): Predicted labels, shape (m, 1).
        
        Returns:
            tuple: A tuple containing:
                - dW (np.array): Gradient w.r.t. weights, shape (n_features, 1).
                - db (float): Gradient w.r.t. bias.
        """
    
        # Your implementation here
        # Hint: Use the formulas above. X.T is the transpose of X.
        # The error term is (y_true - y_pred).
        error = y_true - y_pred
        dW = -2 * np.dot(X.T, error) * y_pred.reshape(-1,) * (1 - y_pred).reshape(-1,)
        dW = np.mean(dW, axis=1).reshape(-1, 1)  # Average over all samples
        db = 0 # ignore bias updates
    
        return dW, db

    return (calculate_gradients,)


@app.function
def update_weights(weights, bias, dW, db, learning_rate):
    """
    Updates the weights and bias using the gradient descent rule.
    
    Args:
        weights (np.array): Current weights.
        bias (float): Current bias.
        dW (np.array): Gradient of weights.
        db (float): Gradient of bias.
        learning_rate (float): The learning rate alpha.
        
    Returns:
        tuple: A tuple containing the updated weights and bias.
    """
    # Your implementation here
    weights = weights - learning_rate*dW
    bias = bias - learning_rate*db
    
    return weights, bias


@app.cell
def _(calculate_gradients, cost_function, forward_pass, initialize_weights):
    def train_neuron(X, y, learning_rate, num_epochs):
        """
        Trains a single neuron using gradient descent.
    
        Args:
            X (np.array): Input data.
            y (np.array): True labels.
            learning_rate (float): The learning rate.
            num_epochs (int): The number of passes through the dataset.
        
        Returns:
            tuple: A tuple containing:
                - trained_weights (np.array): The final learned weights.
                - trained_bias (float): The final learned bias.
                - cost_history (list): A list of cost values at each epoch.
        """
        input_dim = X.shape[1]
    
        # 1. Initialize weights and bias
        weights, bias = initialize_weights(input_dim)
        cost_history = []
    
        # 2. Loop for num_epochs
        for i in range(num_epochs):
            # a. Forward Pass
            z, y_pred = forward_pass(X, weights, bias)
        
            # b. Calculate Cost (and append to history)
            cost = cost_function(y, y_pred)
            cost_history.append(cost)
        
            # c. Calculate Gradients (Backward Pass)
            dW, db = calculate_gradients(X, y, y_pred)
        
            # d. Update Weights
            weights, bias = update_weights(weights, bias, dW, db, learning_rate)
        
            # Optional: Print cost every 100 epochs to check progress
            if i % 100 == 0:
                print(f"Epoch {i}, Cost: {cost}")
            
        return weights, bias, cost_history

    return (train_neuron,)


@app.cell
def _(X, train_neuron, y):
    learning_rate = 0.01
    num_epochs = 10000

    # Train the model
    trained_weights, trained_bias, cost_history = train_neuron(X, y, learning_rate, num_epochs)

    print("\n--- Training Complete ---")
    print(f"Final Weights: \n{trained_weights}")
    print(f"Final Bias: {trained_bias}")
    return (
        cost_history,
        learning_rate,
        num_epochs,
        trained_bias,
        trained_weights,
    )


@app.cell
def _(cost_history, plt):
    # Plot the cost history
    plt.figure(figsize=(8, 6))
    plt.plot(cost_history)
    plt.title('Cost Function Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.show()
    return


@app.cell
def _(forward_pass, np):
    def evaluate_neuron(X, y, trained_weights, trained_bias):
        """
        Evaluates the accuracy of the trained neuron.
    
        Args:
            X (np.array): Input data.
            y (np.array): True labels.
            trained_weights (np.array): The learned weights.
            trained_bias (float): The learned bias.
        
        Returns:
            float: The accuracy of the model (0.0 to 1.0).
        """
        # 1. Get the final predictions from the model using a forward pass
        _, y_pred_proba = forward_pass(X, trained_weights, trained_bias)
    
        # 2. Convert probabilities to binary predictions (0 or 1)
        # Hint: Use a condition like (y_pred_proba >= 0.5)
        # You might want to convert the resulting boolean array to integers (0 or 1).
        y_pred_class = (y_pred_proba >= 0.5)
    
        # 3. Calculate accuracy
        # Hint: Accuracy = (number of correct predictions) / (total number of predictions)
        # np.mean(y_pred_class == y) is a clever way to do this.
        accuracy = np.mean(y_pred_class == y)
    
        return accuracy

    return (evaluate_neuron,)


@app.cell
def _(X, evaluate_neuron, trained_bias, trained_weights, y):
    accuracy = evaluate_neuron(X, y, trained_weights, trained_bias)
    print(f"Final Accuracy on the training set: {accuracy * 100:.2f}%")
    return


@app.cell
def _(plt):
    # now let's test it on a linearly separable dataset
    from sklearn.datasets import make_classification
    # Generate a linearly separable dataset
    X_linear, y_linear = make_classification(n_samples=200, n_features=2,
                                             n_informative=2, n_redundant=0, class_sep=3.0,
                                             n_clusters_per_class=1, random_state=4)
    # Reshape y_linear to be a column vector (m, 1)
    y_linear = y_linear.reshape(-1, 1)
    # Visualize the linearly separable dataset
    plt.figure(figsize=(8, 6))
    plt.scatter(X_linear[:, 0], X_linear[:, 1], c=y_linear, cmap=plt.cm.RdYlBu, edgecolors='k')
    plt.title('Linearly Separable Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()
    return X_linear, y_linear


@app.cell
def _(
    X_linear,
    evaluate_neuron,
    learning_rate,
    num_epochs,
    plt,
    train_neuron,
    y_linear,
):
    # train the neuron on the linearly separable dataset
    trained_weights_linear, trained_bias_linear, cost_history_linear = train_neuron(X_linear, y_linear, learning_rate, num_epochs)

    print("\n--- Training Complete on Linearly Separable Dataset ---")
    print(f"Final Weights: \n{trained_weights_linear}")
    print(f"Final Bias: {trained_bias_linear}")

    # Plot the cost history for the linearly separable dataset
    plt.figure(figsize=(8, 6))
    plt.plot(cost_history_linear)
    plt.title('Cost Function Over Epochs (Linearly Separable Dataset)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)
    plt.show()

    # Evaluate the neuron on the linearly separable dataset
    accuracy_linear = evaluate_neuron(X_linear, y_linear, trained_weights_linear, trained_bias_linear)
    print(f"Final Accuracy on the linearly separable dataset: {accuracy_linear * 100:.2f}%")
    return


if __name__ == "__main__":
    app.run()

