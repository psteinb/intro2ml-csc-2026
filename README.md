# intro2ml-csc-2026

exercises on ML/AI for the CERN School of Computing 2026 edition (adopted from [2025 version](https://github.com/francesco-vaselli/intro2ml-csc-2025) by Francesco Vaselli)

The 4 exercises hours are structured into 3 sets. Each set can provide a exercises sheet on basic concepts and/or a notebook on applications. 

Our suggestion is to try and solve the exercise sheet in you preferred language (pseudocode in Python is provided) and on your local machine. If you are left with more time or you are already familiar with the concepts of the core-concepts part, you can move on to the notebook. 

The notebook for the third set is compute intensive so we can rely on CERN infrastructure for GPUs. You are also free to use the CERN infrastructure for any other exercise before should your local machine not be enough, GPUs have been reserved for us.

## Setting up a notebook on ml.cern.ch (PLEASE NO MORE THAT 1 PERSON PER PAIR)

1. Visit `ml.cern.ch`, login with your CERN account.
2. Select the namespace `csc` from the top left ![step1](/instruction_images/step1.png)
3. Click on `New Notebook` from the quick shortcuts
4. In `Custom Notebook`, select the `kubeflow/kubeflownotebookswg/jupyter-pytorch-cuda-full:v1.9.2` image, to get a pytorch ready image ![step2](/instruction_images/step2.png)
5. When creating a notebook, selecting `CSC - NVIDIA GPU 5GB` as the flavour. For memory and cpu, don't exceed 2 CPUs/10GB memory per notebook. ![step3](/instruction_images/step3.png)
6. In `Advanced Options` -> `Configurations`, deselect `Set EOS as home directory` and `Inject The Oauth2Token` ![step4](/instruction_images/step4.png)

Then you should be able to start the notebook. The machine may take a while before the notebook server is ready to be lunched. 

Once you have the notebook you should be able to simply open a terminal session there and `git clone` this repository to run the exercises.

*Important*: no persistency guaranteed between one session and the other so make sure you save locally something if you'd like to keep it for future reference!