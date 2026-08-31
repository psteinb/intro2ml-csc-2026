# Exploring Neural Networks

## Content

`00_application_bias_variance_tradeoff_exercise.<suffix>`

This folder starts out by introducing the bias/variance tradeoff that we often find, when dealing with machine learning models. This is demonstrated using decision trees. The concept however can be mapped to neural networks too. Solution: `00_application_bias_variance_tradeoff_solutions.<suffix>`

`01_core_concepts_neuron_by_hand.<suffix>`

This exercise helps you make the first steps with torch. The notebook takes you through composing your first neural network using Linear layers and the like. Solution: `01_core_concepts_solutions.<suffix>`

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

