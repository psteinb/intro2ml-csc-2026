# Exploring Neural Networks

## Content

`04_cnns_with_pytorch_exercise.py`

A notebook to dive into a traditional approach to perform classification on images using convolution operation. We use the FashionMNIST dataset to classify images of clothing. This notebook requires you to understand the implications of a convolution on the pixel level and integrate Conv layers into a neural network. Solution: `04_cnns_with_pytorch_solution.py`

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

- free textbook with all fundamentals of modern machine learning (convnets have a dedicated chapter): https://udlbook.github.io/udlbook/
