# Exploring Neural Networks

## Content

`00_decisiontree_sklearn_mechanics.<suffix>`

An exercise to get your fingers warm. This is also a check that your setup is correct. This offers a lightweight entry into classification with decision trees. Note, the task at hand has nothing to do with physics really. This is a common challenge for data scientists, i.e. to work with data they have not helped to produce.

`01_decisiontree_higgs_search_exercise.<suffix>`

This folder starts out by introducing the bias/variance tradeoff that we often find, when dealing with machine learning models. This is demonstrated using decision trees. The concept however can be mapped to neural networks too. Solution: `00_application_bias_variance_tradeoff_solutions.<suffix>`


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

