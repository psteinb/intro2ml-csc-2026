import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise 1: Diving into Classification

    In this notebook, we want to dive into a practical application for machine learning. For this we rely on the `scikit-learn` (`sklearn` for short) library as well as pandas for data handling. The onset is very simple. You are invested for a party and promise to help with the preparations. As you are decorating the buffet with snacks, you mix a bowl of peanuts, walnuts and other chocolate covered candy. Once you are done, the hosts inform you that they expecting guests which are allergic to peanuts. You are now tasked to filter out the peanuts from the bowl of snacks. The dataset, `peanuts.csv` below presents the measurements taken on the bowl of dried snacks. Let's use classification to automate the task.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split

    return DecisionTreeClassifier, np, pd, plot_tree, plt, train_test_split


@app.cell
def _(pd):
    # the data was obtained from https://zenodo.org/records/10014609
    df = pd.read_csv("peanuts.csv")
    return (df,)


@app.cell
def _(df):
    # let's inspect the data
    print(df.shape, "\n", df.dtypes)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the above, we see that most data is detected as `int64`, so this referes to integer numbers. The label however is recognized as object. To convert it into something meaningful, we need to help pandas a bit.
    """)
    return


@app.cell
def _(df):
    df.label = df.label.astype("category")
    df.dtypes
    return


@app.cell
def _(df):
    df.info()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prepare data for training
    """)
    return


@app.cell
def _(df):
    X, y = df[["color","shape", "height","width"]].to_numpy(), df["label"]
    print("input data X is available as:", X.shape, X.dtype)
    print("label data y is available as:", y.shape, y.dtype)
    return X, y


@app.cell
def _(X, train_test_split, y):
    # train_test_split is a super helpful function in sklearn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_test, y_test):
    # inspect the data
    X_test, y_test
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Create the Tree and fit it
    """)
    return


@app.cell
def _(DecisionTreeClassifier, X_train, y_train):
    #see the docs for details https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
    classifier_tree = DecisionTreeClassifier() 
    classifier_tree = classifier_tree.fit(X_train, y_train)
    return (classifier_tree,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## plot the fitted tree
    """)
    return


@app.cell
def _(classifier_tree, plot_tree, plt):
    plt.figure(figsize=(12,12)) #To control the fig size, otherwise is very small
    plot_tree(classifier_tree)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Calculate the Performance Metric
    """)
    return


@app.cell
def _(X_test, classifier_tree, np, y_test):
    y_predict = classifier_tree.predict(X_test)
    acc = 100*np.mean((y_predict==y_test))

    print(f"accuracy is {acc:2.2f} %")
    return


if __name__ == "__main__":
    app.run()

