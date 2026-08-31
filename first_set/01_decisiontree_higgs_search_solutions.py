import marimo

__generated_with = "0.24.0"
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

    This is an fill-in-the-blanks exercise. Go through the notebook and replace all occurances of `...` with the appropriate code.
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

    return DecisionTreeClassifier, pd, plot_tree, plt, sns, train_test_split


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

    # drop rows entries of na in both test and data frames
    data.dropna(inplace=True)
    test.dropna(inplace=True)

    # hint: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.dropna.html
    test.head()
    return


@app.cell
def _(data):
    # prepare features X and labels y for the training set `data`
    # for X use the `data` frame and remove the 'class' column
    # for y use the `data` frame and remove the 'class' column 

    y= data.loc[:, "class"]
    X= data.loc[:, data.columns!='class']
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Signal and Background Comparison
    """)
    return


@app.cell
def _(plt, sns, y):
    # check the frequency of labels present in the training set

    value_counts = y.value_counts()

    sns.barplot(x = value_counts.index, y = value_counts.values)
    plt.title('Label counts')
    plt.show()
    return


@app.cell
def _(plt, sns, test):
    # check the frequency of labels in the test set

    sns.barplot(x = test["class"].value_counts().index, y = test["class"].value_counts().values)
    plt.title('Label counts')
    plt.show()
    return


@app.cell
def _(X, test, train_test_split, y):
    # use the train_test_split function to create a 80-20 split of `data`
    x_train, x_val, y_train, y_val = train_test_split(X, y,test_size = 0.2, # 20% for validation
                                                        random_state = 122)

    # create the test set by dropping the "class" column from `test`
    x_test, y_test = test.loc[:, test.columns!='class'],  test.loc[:, "class"]
    return x_test, x_train, x_val, y_test, y_train, y_val


@app.cell
def _(DecisionTreeClassifier, x_test, x_train, y_train):
    # train the classifier
    tree_classifier = DecisionTreeClassifier()
    tree_classifier = tree_classifier.fit(X=x_train, y= y_train)

    # produce predictions for the training set and the test set
    train_predictions = tree_classifier.predict(x_train)
    test_predictions = tree_classifier.predict(x_test)
    return test_predictions, train_predictions, tree_classifier


@app.cell
def _(test_predictions, train_predictions, y_test, y_train):
    ### Train data accuracy
    from sklearn.metrics import accuracy_score, f1_score

    # print the accuracy scores for the train and the test set
    print("Train data accuracy", accuracy_score(y_train, train_predictions))
    print("Train data accuracy", accuracy_score(y_train, train_predictions))

    print("Test data accuracy ", accuracy_score(y_test, test_predictions))
    return accuracy_score, f1_score


@app.cell
def _(f1_score, test_predictions, train_predictions, y_test, y_train):
    # print the f1 scores for the train and the test set
    print("Train data f1-score for class '1' using f1_score ", f1_score(y_train, train_predictions, pos_label=1))
    print("Train data f1-score for class '0' using f1_score ", f1_score(y_train, train_predictions, pos_label=0))

    print("Test data f1-score for class '1' using f1_score ", f1_score(y_test, test_predictions, pos_label=1))
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
    When the data is imbalanced, meaning the number of observations is different for the different classes
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because the model has learned all of it. Notice that because the model hasn't seen the test data, it's accuracy is lower
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
def _(DecisionTreeClassifier, accuracy_score, x_train, x_val, y_train, y_val):
    acc_val = 0
    ccp_alpha=0.0000
    acc_val_target = 0.685
    while(acc_val < acc_val_target):
      another_tree = DecisionTreeClassifier(ccp_alpha=ccp_alpha)
      another_tree = another_tree.fit(X=x_train, y=y_train)
      val_predictions = another_tree.predict(x_val)
      acc_val = accuracy_score(y_val, val_predictions)
      print(f"Accuracy on validation data: {acc_val}")
      ccp_alpha = ccp_alpha + 0.00001

    print(f"Optimized ccp_alpha: {ccp_alpha}")
    return


if __name__ == "__main__":
    app.run()

