"""Shared matplotlib figures for all training scripts in this folder.

Deliberately free of torch: everything is converted with np.asarray, so the
functions take CPU tensors from the PyTorch scripts as well as the plain Python
lists the micrograd script produces.
"""

import random

import matplotlib

matplotlib.use("Agg")  # save PNGs without needing a display backend

import matplotlib.pyplot as plt
import numpy as np


def denormalize(images, mean, std):
    """Undo Normalize on (N,C,H,W) images so imshow shows the original values."""
    shape = (1, -1, 1, 1)
    mean = np.asarray(mean, dtype=np.float32).reshape(shape)
    std = np.asarray(std, dtype=np.float32).reshape(shape)
    return (np.asarray(images, dtype=np.float32) * std + mean).clip(0, 1)


def plot_loss(history, batch_losses, steps_per_epoch, path, eval_split="test"):
    """Left: loss per batch with a moving average. Right: per-epoch loss and accuracy.

    eval_split names the held-out split in the history dicts, so "test" reads
    the keys test_loss / test_acc and "val" reads val_loss / val_acc.
    """
    fig, (ax_batch, ax_epoch) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax_batch.plot(batch_losses, lw=0.6, alpha=0.3, label="loss per batch")
    window = 50
    if len(batch_losses) >= window:
        moving_avg = np.convolve(batch_losses, np.ones(window) / window, mode="valid")
        ax_batch.plot(range(window - 1, len(batch_losses)), moving_avg, lw=2,
                      label=f"moving average ({window} batches)")
    for epoch in range(1, len(history)):
        ax_batch.axvline(epoch * steps_per_epoch, color="gray", ls=":", lw=0.7)
    ax_batch.set(xlabel="batch step", ylabel="cross-entropy loss",
                 title="Training loss per batch")
    ax_batch.legend()

    epochs = range(1, len(history) + 1)
    ax_epoch.plot(epochs, [h["train_loss"] for h in history], label="train loss")
    ax_epoch.plot(epochs, [h[f"{eval_split}_loss"] for h in history],
                  label=f"{eval_split} loss")
    ax_epoch.set(xlabel="epoch", ylabel="cross-entropy loss",
                 title="Loss and accuracy per epoch")
    ax_epoch.legend(loc="upper left")

    # Accuracy shares the x-axis but needs its own scale.
    ax_acc = ax_epoch.twinx()
    ax_acc.plot(epochs, [h["train_acc"] for h in history], ls="--",
                color="tab:green", label="train accuracy")
    ax_acc.plot(epochs, [h[f"{eval_split}_acc"] for h in history], ls="--",
                color="tab:red", label=f"{eval_split} accuracy")
    ax_acc.set_ylabel("accuracy")
    ax_acc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("written:", path)


def plot_confusion_matrix(matrix, class_names, path, show_counts=True,
                          figsize=(8, 7), tick_fontsize=8, dpi=150):
    """Row-normalised heatmap: every row sums to 1, the diagonal is per-class recall.

    show_counts writes the raw count into every non-empty cell. That fits for
    ten classes and is unreadable for a hundred.
    """
    matrix = np.asarray(matrix)
    normalised = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=figsize)
    image = ax.imshow(normalised, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, label="fraction of the true class")

    if show_counts:
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                count = int(matrix[i, j])
                if count == 0:
                    continue
                ax.text(j, i, str(count), ha="center", va="center", fontsize=6,
                        color="black" if normalised[i, j] > 0.5 else "white")

    # Long class names only fit on the x-axis when they are turned upright.
    rotation = 90 if max(len(name) for name in class_names) > 2 else 0
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=rotation, fontsize=tick_fontsize)
    ax.set_yticklabels(class_names, fontsize=tick_fontsize)
    ax.set(xlabel="predicted class", ylabel="true class",
           title="Confusion matrix (row-normalised, counts in cells)" if show_counts
           else "Confusion matrix (row-normalised)")

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    print("written:", path)


def plot_misclassified(images, labels, preds, class_names, mean, std, path, n=64,
                       seed=0, figsize=(10, 11), title="{true} -> {pred}",
                       title_fontsize=8):
    """Show n randomly sampled errors in an 8x8 grid, titled with truth and prediction.

    images are normalised (N,C,H,W); one channel is drawn in grey, three as RGB.
    """
    if len(images) == 0:
        print("no misclassified images - skipping plot")
        return

    sample = random.Random(seed).sample(range(len(images)), min(n, len(images)))
    shown = denormalize(np.asarray(images, dtype=np.float32)[sample], mean, std)
    # (C,H,W) -> (H,W,C), because that is the layout imshow expects.
    shown = shown.transpose(0, 2, 3, 1)
    grey = shown.shape[-1] == 1

    fig, axes = plt.subplots(8, 8, figsize=figsize)
    for ax in axes.flat:
        ax.axis("off")

    for ax, idx, image in zip(axes.flat, sample, shown):
        # "nearest" keeps the pixels as hard blocks instead of blurring them:
        # the images have no more detail than this, so smoothing only invents it.
        if grey:
            ax.imshow(image[..., 0], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        else:
            ax.imshow(image, interpolation="nearest")
        ax.set_title(title.format(true=class_names[int(labels[idx])],
                                  pred=class_names[int(preds[idx])]),
                     fontsize=title_fontsize, color="firebrick")

    fig.suptitle(f"{len(shown)} of {len(images)} misclassified test images", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("written:", path)
