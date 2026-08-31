import marimo

__generated_with = "0.23.16"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import subprocess

    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exercise 2: Finding the Higgs Boson

    This is an fill-in-the-blanks exercise. Go through the notebook and replace all occurances of `...` with the appropriate code. The comments will guide you what to do. If need be, use the previous notebook to help you.
    """)
    return


@app.cell
def _(subprocess):
    # the following lines will only work on Linux and Mac
    #! wget "https://github.com/astrocronopio/datasets/raw/main/higgs_dataset.zip"
    subprocess.call(['wget', 'https://github.com/astrocronopio/datasets/raw/main/higgs_dataset.zip'])
    #! unzip -qq -o higgs_dataset.zip
    subprocess.call(['unzip', '-qq', '-o', 'higgs_dataset.zip'])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Importing required libraries
    """)
    return


@app.cell
def _():
    import os
    import numpy as np
    import pandas as pd
    import seaborn as sns

    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt

    return DecisionTreeClassifier, pd, plot_tree, plt, sns


@app.cell
def _(pd):
    data = pd.read_csv("./train.csv",na_values="?")
    test = pd.read_csv("./test.csv", na_values="?")
    print("Train Data Shape:", data.shape)
    print("Test Data Shape:", test.shape)

    # This is to see the first lines of the train data
    data.head()
    return data, test


@app.cell
def _(data, test):
    # drop the column in both `test` and `data` frames

    data.drop(["id"], axis = 1, inplace=True)
    test.drop(["id"], axis = 1, inplace=True)

    # drop rows which contain entries of na in both test and data frames
    ...

    # hint: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
    test.head()
    return


@app.cell
def _():
    # prepare features X and labels y for the training set `data`
    # our prediction target will be the column `class` from the `data` table

    # for X: use the `data` frame without the `class` column 
    # for y: use only the `class` column of the `data` frame

    ...
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Signal and Background Comparison
    """)
    return


@app.cell
def _(plt, sns):
    # check the frequency of labels present in the training set

    value_counts = ...

    sns.barplot(x = value_counts.index, y = value_counts.values)
    plt.title('Label counts')
    plt.show()
    return


@app.cell
def _():
    # check the frequency of labels in the test set

    ...
    return


@app.cell
def _():
    # use the train_test_split function to create a 80-20 split of `data`
    x_train, x_val, y_train, y_val = ..., ..., ..., ...

    # create the test set by dropping the "class" column from `test`
    x_test, y_test = ..., ...
    return x_test, x_train, x_val, y_test, y_train, y_val


@app.cell
def _(DecisionTreeClassifier, x_test, x_train, y_train):
    # train the classifier
    tree_classifier = DecisionTreeClassifier()
    tree_classifier = tree_classifier.fit(X=x_train, y= y_train)

    # produce predictions for the training set and the test set
    train_predictions = tree_classifier.predict(x_train)
    test_predictions = tree_classifier.predict(x_test)
    return (tree_classifier,)


@app.cell
def _():
    ### Train data accuracy
    from sklearn.metrics import accuracy_score, f1_score

    # print the accuracy scores for the train and the test set
    return


@app.cell
def _():
    # print the f1 scores for the train and the test set
    ...
    return


@app.cell
def _(plot_tree, plt, tree_classifier):
    plt.figure(figsize=(20,14)) #To control the fig size, otherwise is very small
    plot_tree(tree_classifier, max_depth=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Questions

    - In what cases is the accuracy score not a good measure of our data?

    - Why is the accuracy for train 100%?
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pruning

    We saw that test accuracy reaches 64%. But we would like to go higher. Take the `ccp_alpha` parameter of the decision tree object and increase it by 1e-5 until you reach an accuracy of 68.5%!
    """)
    return


@app.cell
def _():
    acc_val = 0
    ccp_alpha=0.0000
    acc_val_target = 0.685
    while(acc_val < acc_val_target):
      ...

    print(f"Optimized ccp_alpha: {ccp_alpha}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus Exercise

    Swap the training above with a boosted decision tree of a bagging decision tree. What do you see with respect to performance improvements?
    """)
    return


@app.cell
def _():
    ...
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
