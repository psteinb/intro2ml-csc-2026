# Introduction to Machine Learning, CSC 2026

Exercises on ML/AI for the CERN School of Computing 2026 edition (adopted from [2025 version](https://github.com/francesco-vaselli/intro2ml-csc-2025) by Francesco Vaselli).

The 4 exercise hours are structured into 4 folders available in this repository. Each directory provides exercise notebooks for students to work with. If you are stuck or tried really hard on a problem for 5-10 minutes, feel free to consult the adjoint solution notebooks. Note, running only the solution notebooks will impair your learning experience.

Our suggestion is to try and solve the exercise notebooks on CERN compute infrastructure or on your local machine. If you are left with more time or you are already familiar with the concepts of the core-concepts part, you can move on to the next notebook or consider the bonus notebooks. 

The notebook for the last folder `exercise04` is compute intensive. Please use CERN's swan infrastructure for GPU access. You are also free to use the CERN infrastructure for any other exercise should your local machine not be enough. **Note, support for local installation issues is limited during exercise sessions.**

**Important notice for marimo fans**: The Swan (experimental) jupyterlab interface offers you to execute jupyter notebooks, but also provides functionality to execute marimo notebooks. For this reason, every jupyter notebook has an adjoint marimo notebook next to it. However, at the time of writing the marimo environment doesn't pick up the CERN software stack. Use the package manager in marimo to install dependencies according to the `requirements.txt` in every folder.

## Set up with SWAN

### Enter [SWAN](https://swan.cern.ch)

![](instruction_images/csc26-swan01-stack109-cu125_small.png)

Authenticate with your csc26 account.

### Download the git repository [github.com/psteinb/intro2ml-csc-2026](https://github.com/psteinb/intro2ml-csc-2026)

![](instruction_images/csc26-swan02-clonerepo_small.png)

### Insert Repo URL

![](instruction_images/csc26-swan03-repourl_small.png)

### Confirm Download

![](instruction_images/csc26-swan04-confirm-gitrepo_small.png)

### The code is now downloaded

![](instruction_images/csc26-swan05-repo-downloaded_small.png)

## Curriculum

- Mon, Aug 31
    - Lecture 1, 11:30am -> Course1 material
    - Lecture 2, 12:30pm -> Course2 material 
- Tue, Sep 01
    - Exercise 1, 12:30pm -> Intro DecisionTrees (Peanuts and Higgs Search) in [exercise01](exercise01/)
    - Exercise 2, 15:30pm -> Bias Variance Tradeoff in [exercise02](exercise02/)
- Wed, Sep 02
    - Lecture 3, 8:45am -> Course3 material
    - Exercise 3, 9:45am -> MLPs in pytorch (backprop by hand, train an MLP) [exercise03](exercise03/)
    - Lecture 4, 2:30pm -> Course4 material
    - Exercise 4, 4:00pm -> CNNs [exercise04](exercise04/)
- Thu, Sep 03
    - Lecture 5, 2:30pm -> Course5 material
