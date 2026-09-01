import marimo

__generated_with = "0.24.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import torch
    from matplotlib import pyplot as plt
    from typing import Tuple

    torch.manual_seed(41)
    return Tuple, mo, plt, torch


@app.cell
def _(Tuple, torch):
    def make_dataset(
        num_samples: int = 10000,
        length: int = 64,
        signal_width: int = 4,
        num_regions: int = 4,
        signal_height: float = 0.8,
        noise_scale: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create noisy 1D signals and a binary mask for their signal regions."""
        region_width = length // num_regions
        values = torch.distributions.HalfNormal(noise_scale).sample((num_samples, length))
        masks = torch.zeros(num_samples, length)

        for sample in range(num_samples):
            for region in range(num_regions):
                start = region * region_width + torch.randint(
                    region_width - signal_width + 1, ()
                ).item()
                masks[sample, start : start + signal_width] = 1.0

        return values + signal_height * masks, masks

    return (make_dataset,)


@app.cell
def _(make_dataset):
    X_train, y_train = make_dataset(4000)
    X_test, y_test = make_dataset(400)
    return X_test, X_train, y_test, y_train


@app.cell
def _(X_train, plt, y_train):
    data_figure, data_axes = plt.subplots(1, 4, figsize=(14, 3))
    for data_index, data_axis in enumerate(data_axes):
        data_axis.plot(X_train[data_index], label="noisy signal")
        data_axis.plot(y_train[data_index], label="target mask")
        data_axis.set_title(f"sample {data_index}")
        data_axis.set_ylim(-0.05, 1.3)
    data_axes[-1].legend()
    data_figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Bonus: From CNNs to U-Nets

    The CNN exercise classified a complete image with one label. Here the task is
    **segmentation**: for every position in a noisy 1D signal, predict whether it
    belongs to a signal region. Therefore an input of shape `[batch, 1, 64]` must
    produce one prediction for each of the 64 positions.

    A **U-Net** is an encoder-decoder convolutional architecture introduced for
    biomedical image segmentation by [Ronneberger, Fischer, and Brox (2015)](https://arxiv.org/abs/1505.04597).
    The encoder uses convolutions and downsampling to combine information from a
    wider neighbourhood. The decoder uses transpose convolutions to restore the
    original resolution. At every resolution, a **skip connection** concatenates
    encoder features with decoder features so that fine location information is
    not lost during downsampling.

    This notebook uses the same idea for 1D signals: `Conv1d` replaces `Conv2d`,
    but the axes still mean `[batch, channels, length]`. The goal is not to
    memorise U-Net variants; use the shape table below to construct one small,
    working example.

    ## Architecture plan

    Starting with `[batch, 1, 64]`, use eight feature channels throughout:

    | Stage | Operation | Output shape |
    | --- | --- | --- |
    | head | convolution, padding 1 | `[batch, 8, 64]` |
    | encoder 1 | convolution, stride 2, padding 1 | `[batch, 8, 32]` |
    | encoder 2 | convolution, stride 2, padding 1 | `[batch, 8, 16]` |
    | bottleneck | convolution, stride 2, padding 1 | `[batch, 8, 8]` |
    | decoder 1 | transpose convolution | `[batch, 8, 16]` |
    | skip 1 | concatenate with encoder 2 along `dim=1` | `[batch, 16, 16]` |
    | decoder 2 | restore then concatenate with encoder 1 | `[batch, 16, 32]` |
    | decoder 3 | restore then concatenate with head | `[batch, 16, 64]` |
    | output | convolution to one channel | `[batch, 1, 64]` |

    Use raw output **logits** in the model. `BCEWithLogitsLoss` combines the
    sigmoid conversion and binary cross-entropy safely; apply `torch.sigmoid`
    only when displaying predictions or converting them to binary masks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exercise

    1. Complete the layers marked `TODO`. Use `kernel_size=3`, `padding=1`, and
       `ReLU` after each feature-building convolution.
    2. Complete the `forward` method, saving encoder outputs for the three skip
       connections. Concatenate channels with `torch.cat(..., dim=1)`.
    3. Run the shape check before training. If it fails, compare each intermediate
       shape with the architecture table.
    4. Train the model and inspect the probability and binary-mask plots.
    """)
    return


@app.cell
def _(torch):
    class UNet1d(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Sequential(
                torch.nn.Conv1d(1, 8, kernel_size=3, padding=1), torch.nn.ReLU()
            )
            # TODO: create three stride-2 encoder blocks: 8 channels to 8 channels.
            self.down1 = ...
            self.down2 = ...
            self.down3 = ...
            # TODO: create three transpose convolutions that double the length.
            self.up1 = ...
            self.up2 = ...
            self.up3 = ...
            # TODO: after each skip concatenation, reduce 16 channels back to 8.
            self.merge1 = ...
            self.merge2 = ...
            self.output = torch.nn.Conv1d(16, 1, kernel_size=3, padding=1)

        def forward(self, x):
            head = self.head(x)
            # TODO: compute encoder1, encoder2, and bottleneck.
            # TODO: upsample, concatenate the matching encoder output, and merge.
            # TODO: return raw logits with shape [batch, 1, 64].
            ...

    return (UNet1d,)


@app.cell
def _(UNet1d, torch):
    model = UNet1d()
    probe = torch.zeros(2, 1, 64)
    output = model(probe)
    assert output.shape == (2, 1, 64), output.shape
    return (model,)


@app.cell
def _(X_test, X_train, torch, y_test, y_train):
    batch_size = 64
    train_dataset = torch.utils.data.TensorDataset(X_train.unsqueeze(1), y_train.unsqueeze(1))
    test_dataset = torch.utils.data.TensorDataset(X_test.unsqueeze(1), y_test.unsqueeze(1))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    return test_loader, train_loader


@app.cell
def _(model, test_loader, torch, train_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = torch.nn.BCEWithLogitsLoss()
    history = {"train_loss": [], "test_loss": []}

    for epoch in range(10):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(inputs), targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(inputs)
        history["train_loss"].append(train_loss / len(train_loader.dataset))

        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for inputs, targets in test_loader:
                test_loss += loss_function(model(inputs), targets).item() * len(inputs)
        history["test_loss"].append(test_loss / len(test_loader.dataset))

        print(f"epoch {epoch + 1:02d}: train={history['train_loss'][-1]:.4f}, test={history['test_loss'][-1]:.4f}")
    return history, loss_function


@app.cell
def _(model, test_loader, torch):
    display_inputs, display_targets = next(iter(test_loader))
    model.eval()
    with torch.no_grad():
        display_probabilities = torch.sigmoid(model(display_inputs))
    display_predictions = display_probabilities >= 0.5
    position_accuracy = (display_predictions == display_targets.bool()).float().mean()
    print(f"per-position accuracy: {position_accuracy:.1%}")
    return display_inputs, display_predictions, display_probabilities, display_targets


@app.cell
def _(display_inputs, display_predictions, display_probabilities, display_targets, plt):
    prediction_figure, prediction_axes = plt.subplots(1, 4, figsize=(14, 3))
    for prediction_index, prediction_axis in enumerate(prediction_axes):
        prediction_axis.plot(display_inputs[prediction_index, 0], label="input")
        prediction_axis.plot(display_targets[prediction_index, 0], label="target")
        prediction_axis.plot(display_probabilities[prediction_index, 0], label="probability")
        prediction_axis.plot(display_predictions[prediction_index, 0], "--", label="prediction")
        prediction_axis.set_ylim(-0.05, 1.3)
    prediction_axes[-1].legend()
    prediction_figure
    return


if __name__ == "__main__":
    app.run()
