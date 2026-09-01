import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import subprocess

    import matplotlib.pyplot as plt
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import (
        KFold,
        StratifiedKFold,
        cross_validate,
        train_test_split,
    )
    from sklearn.tree import DecisionTreeClassifier

    return (
        DecisionTreeClassifier,
        HistGradientBoostingClassifier,
        KFold,
        RandomForestClassifier,
        StratifiedKFold,
        accuracy_score,
        cross_validate,
        mo,
        pd,
        plt,
        sns,
        subprocess,
        train_test_split,
    )


@app.cell
def _(subprocess):
    # Download the dataset if it is not already available. This command is for Linux and macOS.
    subprocess.call(["wget", "-nc", "https://github.com/astrocronopio/datasets/raw/main/higgs_dataset.zip"])
    subprocess.call(["unzip", "-qq", "-o", "higgs_dataset.zip"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bonus exercise: Cross-validation for the Higgs search

    A measured test accuracy depends on which events happen to be held out. We first
    repeat the familiar Higgs decision-tree task with three train/test splits. Then we
    replace one split by 15 validation folds, first without and then with
    stratification.
    """)
    return


@app.cell
def _(pd):
    data = pd.read_csv("./train.csv", na_values="?")
    data = data.drop(columns="id").dropna()
    X = data.drop(columns="class")
    y = data["class"]
    print("Usable events:", len(data))
    print("Signal/background counts:")
    print(y.value_counts().sort_index())
    return X, y


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 1: Three train/test splits

    Let's explore the results of the previous notebook a bit.

    Train exactly the same decision tree three times. Only the seed supplied to
    `train_test_split` should change. For every training, print the test accuracy! Are they are all the same number?
    """)
    return


@app.cell
def _(DecisionTreeClassifier, X, accuracy_score, train_test_split, y):
    split_scores = []
    for seed in (12, 42, 99):
        _x_train, _x_test, _y_train, _y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        _classifier = DecisionTreeClassifier(max_depth=5, random_state=0)
        _classifier.fit(_x_train, _y_train)
        test_accuracy = accuracy_score(_y_test, _classifier.predict(_x_test))
        split_scores.append(test_accuracy)
        print(f"Seed {seed}: test accuracy = {test_accuracy:.3f}, first 5 labels {_y_train[:5]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    As you can tell, we don't see three identical accuracy values. The deviation is not large, but it is clearly present. This effect is caused from the mere effect that by randomly splitting a dataset, we fix the training set. With this and given the fact that the decision tree classifier has limited predictive capacity, the prediction results must vary stochastically. Hence, the accuracy is a random variable, because the predictions are random variables.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 2: Regular 15-fold cross-validation

    `sklearn` provides facilities to expand on this effect. In the next cells, we will explore cross-validation. This method splits a dataset multiple times at best in non-overlapping folds. The object to establish these splits is called `KFold`. See [here](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html) for details.

    Use the KFold object to split the dataset into 15 folds. Store the test accuracy scores into a list so that you create a histogram later.
    """)
    return


@app.cell
def _(KFold):
    regular_cv = KFold(n_splits=15, shuffle=True, random_state=42)
    return (regular_cv,)


@app.cell
def _(DecisionTreeClassifier, X, accuracy_score, regular_cv, y):
    regular_scores = []
    for _fold, (_train_indices, _validation_indices) in enumerate(
        regular_cv.split(X), start=1
    ):
        _x_train = X.iloc[_train_indices]
        _x_validation = X.iloc[_validation_indices]
        _y_train = y.iloc[_train_indices]
        _y_validation = y.iloc[_validation_indices]

        if _fold <= 3:
            print(f"Regular fold {_fold} validation label counts:")
            print(_y_validation.value_counts().sort_index())

        _classifier = DecisionTreeClassifier(max_depth=5, random_state=0)
        _classifier.fit(_x_train, _y_train)
        regular_scores.append(
            accuracy_score(_y_validation, _classifier.predict(_x_validation))
        )

    print(f"Regular CV mean accuracy: {sum(regular_scores) / len(regular_scores):.3f}")
    return (regular_scores,)


@app.cell
def _(plt, regular_scores):
    plt.plot(range(1, 16), regular_scores, marker="o")
    plt.xlabel("Validation fold")
    plt.ylabel("Validation accuracy")
    plt.title("Accuracy variation across regular 15-fold CV")
    plt.xticks(range(1, 16))
    plt.ylim(0.6, 0.7)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We again observe a variation in the test set accuracy for each fold.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 3: Stratified 15-fold cross-validation

    `StratifiedKFold` uses the labels while assigning events to folds. Each validation
    fold should therefore have nearly the same signal/background fraction as the full
    dataset. Compare the first three label counts and the spread of accuracies with
    the regular folds above.
    """)
    return


@app.cell
def _(StratifiedKFold):
    stratified_cv = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
    return (stratified_cv,)


@app.cell
def _(DecisionTreeClassifier, X, accuracy_score, stratified_cv, y):
    stratified_scores = []
    for _fold, (_train_indices, _validation_indices) in enumerate(
        stratified_cv.split(X, y), start=1
    ):
        _x_train = X.iloc[_train_indices]
        _x_validation = X.iloc[_validation_indices]
        _y_train = y.iloc[_train_indices]
        _y_validation = y.iloc[_validation_indices]

        if _fold <= 3:
            print(f"Stratified fold {_fold} validation label counts:")
            print(_y_validation.value_counts().sort_index())

        _classifier = DecisionTreeClassifier(max_depth=5, random_state=0)
        _classifier.fit(_x_train, _y_train)
        stratified_scores.append(
            accuracy_score(_y_validation, _classifier.predict(_x_validation))
        )

    print(f"Stratified CV mean accuracy: {sum(stratified_scores) / len(stratified_scores):.3f}")
    return (stratified_scores,)


@app.cell
def _(pd, regular_scores, stratified_scores):
    summary = pd.DataFrame(
        {
            "regular KFold": regular_scores,
            "stratified KFold": stratified_scores,
        }
    ).agg(["mean", "std", "min", "max"])
    print("Validation accuracy summary:")
    print(summary)
    return


@app.cell
def _(plt, regular_scores, stratified_scores):
    plt.plot(range(1, 16), regular_scores, marker="o", label="regular KFold")
    plt.plot(range(1, 16), stratified_scores, marker="o", label="stratified KFold")
    plt.xlabel("Validation fold")
    plt.ylabel("Validation accuracy")
    plt.title("Regular and stratified 15-fold validation accuracy")
    plt.xticks(range(1, 16))
    plt.ylim(0.6, 0.7)
    plt.legend()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the stratified cross validation, the deviation of the classification score is smaller. The effect is not strong here as the original dataset is not really imblanaced, so stratification versus blind uniform splits has only mild effects.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Part 4: Compare three classifiers with stratified CV

    Now that the split procedure is fixed, compare the decision tree with the random
    forest and gradient-boosting classifiers from the earlier version of this exercise.
    Each model is trained and evaluated on the same 15 stratified folds, so their score
    distributions can be compared fairly.
    """)
    return


@app.cell
def _(
    DecisionTreeClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
):
    classifiers = {
        "decision tree": DecisionTreeClassifier(max_depth=5, random_state=0),
        "random forest": RandomForestClassifier(
            n_estimators=50, max_depth=8, n_jobs=-1, random_state=0
        ),
        "gradient boosting": HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, random_state=0
        ),
    }
    return (classifiers,)


@app.cell
def _(X, classifiers, cross_validate, pd, stratified_cv, y):
    classifier_scores = {}
    for name, _classifier in classifiers.items():
        result = cross_validate(
            _classifier,
            X,
            y,
            cv=stratified_cv,
            scoring="accuracy",
            n_jobs=-1,
            return_train_score=False,
        )
        classifier_scores[name] = result["test_score"]

    final_scores = pd.DataFrame(classifier_scores)
    final_scores
    return (final_scores,)


@app.cell
def _(final_scores):
    final_scores.agg(["mean", "std", "min", "max"]).T.sort_values("mean", ascending=False)
    return


@app.cell
def _(final_scores, plt, sns):
    sns.boxplot(data=final_scores)
    plt.ylabel("Stratified cross-validation accuracy")
    plt.title("Classifier performance across 15 held-out folds")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Interpretation

    The three train/test scores in the beginning demonstrated that one split gives only one possible trained model and one performance estimate. Regular cross-validation gives 15 estimates, but its folds can have different class compositions and thus be unrepresentative. Stratification makes the class counts in the validation folds almost identical and typically makes this accuracy assessment more stable.

    Compare the classifier means together with their spreads rather than selecting a
    model from one lucky fold. Use this cross-validation result to choose a model and
    its hyperparameters. Keep `test.csv` untouched until that choice is complete, then
    use it once for the final performance estimate. Take note, that in the box plot above the deviation (signalled by the whiskers) demos that the performance results for randomforest and gradient boosting are clearly superior than the decision tree. However, the "error bars" for random forest and gradient boosting do overlap. Thus, we would need more data and more folds to take definite decision which model is better for this task!
    """)
    return


if __name__ == "__main__":
    app.run()
