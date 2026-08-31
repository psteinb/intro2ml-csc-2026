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
    # CNNs with pytorch

    To demonstrate the use of Convolutional Neural Networks with Pytorch, we will use an intersting dataset by the name of FashionMNIST. Derived by the spirits of the original [MNIST](https://en.wikipedia.org/wiki/MNIST_database), the fashion reseller Zalando donated a large cohort of product images to the communities for teaching Deep Learning to beginners. See also the [original repository](https://github.com/zalandoresearch/fashion-mnist) of the dataset. `pytorch` can be paired by a `torch` compatible library called `torchvision` which offers several vision related building blocks and datasets.

    ## Loading FashionMNIST
    """)
    return


@app.cell
def _():
    from torchvision.datasets import FashionMNIST
    from torchvision import transforms

    return FashionMNIST, transforms


@app.cell
def _(transforms):
    transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])
    return (transform,)


@app.cell
def _(FashionMNIST, transform):
    train_ds = FashionMNIST(root=".",train=True, download=True, transform=transform)
    return (train_ds,)


@app.cell
def _(FashionMNIST, transform):
    test_ds = FashionMNIST(root=".",train=False, download=True, transform=transform)
    return (test_ds,)


@app.cell
def _(train_ds):
    # check how many samples are contained
    print(len(train_ds))
    return


@app.cell
def _(test_ds):
    print(len(test_ds))
    return


@app.cell
def _(train_ds):
    # inspect the first training image
    first_X, first_y = train_ds[0]
    return first_X, first_y


@app.cell
def _(first_X, first_y):
    print("X",first_X.shape, first_X.dtype)
    print("y",first_y.shape, first_y.dtype, f"label={first_y.item()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    From the above we learn, that each image is build from 28x28 pixels stored as grey value intensities (as there is only one channel). Each pixel is encoded as `float32`. Further, the labels are encoded as `int64`. Let's try to plot them:
    """)
    return


@app.cell
def _(train_ds):
    import matplotlib.pyplot as plt

    fig,ax = plt.subplots(1,4,figsize=(16,5))

    for p in range(4):
        X,y = train_ds[p]
        ax[p].imshow(X[0,...])
        ax[p].set_title(f"FashionMNIST label={y.item()}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Defining the model
    """)
    return


@app.cell
def _():
    from torch import nn
    import torch.nn.functional as F

    return (nn,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Write a CNN class which consists of:
    - a 3x3 convolution with 32 output channels with 1 input channel
    - a 3x3 convolution with 32 input channels and 64 output channels
    - one dropout layer with rate 0.25
    - one dropout layer with rate 0.5
    - one linear layers taking going from 9216 inputs to 128 outputs
    - one linear layer with 128 inputs and 10 outputs
    """)
    return


@app.cell
def _(nn):
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            ...
        
        def forward(self, x):
            ...

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preparing the training loop
    """)
    return


@app.cell
def _():
    import torch

    batch_size = 32
    learning_rate = .01
    return (torch,)


app._unparsable_cell(
    r"""
    train_loader = torch.utils.data.DataLoader(train_ds,batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_ds,batch_size=batch_size)

    # fill in the blanks!
    model = 
    optimizer = torch.optim.Adadelta(...)
    """,
    name="_"
)


@app.cell
def _(loss):
    def train(args, model, train_loader, optimizer, epoch):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data, target
            optimizer.zero_grad()

            #fill in the blanks
            ...
        
            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
                if args.dry_run:
                    break

    return (train,)


@app.cell
def _(pred, torch):
    def test(model, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data, target
                # fill in the blanks
                ...
            
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

    return (test,)


@app.cell
def _():
    from types import SimpleNamespace

    args_ = dict(dry_run = False, log_interval=100, epochs=15)
    args = SimpleNamespace(**args_)
    args
    return (args,)


@app.cell
def _(args, model, optimizer, test, test_loader, train, train_loader):
    for epoch in range(1, args.epochs + 1):
        train(args, model, train_loader, optimizer, epoch)
        test(model, test_loader)
    return


@app.cell
def _():
    # plot the training and test loss!
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can observe several things from the notebook above:
    - convolutions on 28x28 pixel images can be very slow on a CPU
    - the network exposes very good accuracy right from the start (85%)
    - the flow of the network is similar to the MLP, only that we use Conv+Pool operations instead of linear layers
    - we use the same loss function
    """)
    return


if __name__ == "__main__":
    app.run()

