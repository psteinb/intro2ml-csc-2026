# Exploring Neural Networks

## Content

`02_application_bias_variance_tradeoff_exercise.<suffix>`

This notebook starts out by introducing the bias/variance tradeoff that we often find, when dealing with machine learning models. This is demonstrated using decision trees. The concept however can be mapped to neural networks too. Solution: `02_application_bias_variance_tradeoff_solutions.<suffix>`

`b2_classifier_crossval_exercise.py`

In the bias variance tradeoff exercise, we embarked on an activity that AI scientists and engineers face regularly: which model architecture is best for the task? This notebook picks up the higgs search again and compares decision-tree, random-forest, and gradient-boosting classifiers with cross-validation. This notebook demonstrates: measured model performance varies like a random variable - the data split and the limits of the dataset in the firstplace induces randomness. Solution: `b2_classifier_crossval_solution.py`.

## Installation

The following material was tested on SWAN using ... in August 2026. If you would like to run the material locally, you can use the `requirements.txt` file to install software dependencies. In order to be more flexible, use the `uv` tool (see [here](https://docs.astral.sh/uv/) for details).

```bash
uv venv --system-site-packages
source .venv/bin/activate
uv pip install -r ./requirements.txt
```

The instructions above assume that you have a running jupyter lab or marimo infrastructure inplace. If not, install the following in the same environment as you installed requirements!

```bash
source .venv/bin/activate
uv pip install jupyterlab 
#uv pip install marimo #if you like this environment better
```

## Further Reading

- nice tutorial by the `sklearn` developers: https://inria.github.io/scikit-learn-mooc/ (ready for self-study)
- very good overview paper about implications of stochastic learning: https://arxiv.org/abs/1811.12808 
- benchmarking AI systems as a scientific discipline: https://www.mlbenchmarks.org/
