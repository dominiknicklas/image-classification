"""MNIST: small convolutional network, same training setup as the MLP baseline.

The only difference to mnist_mlp.py is the model: two conv blocks with max
pooling instead of a single hidden layer. Optimizer, learning rate, batch size
and epoch count are identical, so the accuracy gap is attributable to the
architecture. Architecture and hyperparameters are unchanged from the original
version; only the structure of the script and the evaluation were reworked.

Writes best.pt and three PNGs to ../results/.
"""

import matplotlib

matplotlib.use("Agg")  # save PNGs without needing a display backend

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ----------------------------------------------------------------------
# Hyperparameters - the only place to tune
# ----------------------------------------------------------------------
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.1
CHANNELS = 1
NUM_WORKERS = 4          # set to 0 when running inside Jupyter
SEED = 0
DATA_DIR = "data"
OUT_DIR = "../results"
PREFIX = "mnist_cnn"


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def channel_stats(dataset, channels):
    """Return per-channel mean and standard deviation over a tensor dataset."""
    total = torch.zeros(channels)
    total_sq = torch.zeros(channels)
    n = 0

    for images, _ in DataLoader(dataset, batch_size=1000):
        total += images.sum(dim=[0, 2, 3])
        total_sq += (images**2).sum(dim=[0, 2, 3])
        n += images.shape[0] * images.shape[2] * images.shape[3]

    mean = total / n
    std = (total_sq / n - mean**2).sqrt()
    return mean, std


def build_model(num_classes):
    """Two conv-pool blocks, then the same classifier head as the MLP."""
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),    # (B,1,28,28) -> (B,32,28,28)
        nn.ReLU(),
        nn.MaxPool2d(2),                   #             -> (B,32,14,14)

        nn.Conv2d(32, 64, 3, padding=1),   #             -> (B,64,14,14)
        nn.ReLU(),
        nn.MaxPool2d(2),                   #             -> (B,64,7,7)

        nn.Flatten(),                      #             -> (B,3136)
        nn.Linear(64 * 7 * 7, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),       # logits, no softmax
    )


def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """Run one pass over the training data. Returns loss, accuracy, per-batch losses."""
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    batch_losses = []

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.item())
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, correct / total, batch_losses


@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device):
    """Return average loss and accuracy over a dataloader."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += loss_fn(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    return loss_sum / total, correct / total


@torch.no_grad()
def collect_misclassified(model, dataloader, device):
    """Collect every wrong prediction as (images, true labels, predictions)."""
    model.eval()
    images_out, labels_out, preds_out = [], [], []

    for images, labels in dataloader:
        preds = model(images.to(device)).argmax(dim=1).cpu()
        wrong = preds != labels
        images_out.append(images[wrong])
        labels_out.append(labels[wrong])
        preds_out.append(preds[wrong])

    return torch.cat(images_out), torch.cat(labels_out), torch.cat(preds_out)


@torch.no_grad()
def confusion_matrix(model, dataloader, device, num_classes):
    """Count predictions per (true class, predicted class) pair.

    Row i, column j holds how often an image of class i was predicted as
    class j. The diagonal are the correct ones, everything else is an error.
    """
    model.eval()
    matrix = torch.zeros(num_classes, num_classes, dtype=torch.long)

    for images, labels in dataloader:
        preds = model(images.to(device)).argmax(dim=1).cpu()
        # Flatten each (true, pred) pair into a single index, then count.
        flat = labels * num_classes + preds
        matrix += torch.bincount(flat, minlength=num_classes**2).reshape(num_classes, num_classes)

    return matrix


def print_top_confusions(matrix, class_names, n=10):
    """List the class pairs the model mixes up most often."""
    errors = matrix.clone()
    errors.fill_diagonal_(0)  # correct predictions are not confusions
    top = errors.flatten().topk(n)

    print(f"\ntop {n} confusions (count | true -> predicted):")
    for count, index in zip(top.values.tolist(), top.indices.tolist()):
        true_idx, pred_idx = divmod(index, len(class_names))
        print(f"  {count:4d} | {class_names[true_idx]:>10} -> {class_names[pred_idx]}")


def denormalize(images, mean, std):
    """Undo Normalize so imshow shows the original grey values."""
    shape = (1, -1, 1, 1)
    return (images * std.view(shape) + mean.view(shape)).clamp(0, 1)


def plot_loss(history, batch_losses, steps_per_epoch, path):
    """Left: loss per batch with a moving average. Right: per-epoch loss and accuracy."""
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
    ax_epoch.plot(epochs, [h["test_loss"] for h in history], label="test loss")
    ax_epoch.set(xlabel="epoch", ylabel="cross-entropy loss",
                 title="Loss and accuracy per epoch")
    ax_epoch.legend(loc="upper left")

    # Accuracy shares the x-axis but needs its own scale.
    ax_acc = ax_epoch.twinx()
    ax_acc.plot(epochs, [h["train_acc"] for h in history], ls="--",
                color="tab:green", label="train accuracy")
    ax_acc.plot(epochs, [h["test_acc"] for h in history], ls="--",
                color="tab:red", label="test accuracy")
    ax_acc.set_ylabel("accuracy")
    ax_acc.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("written:", path)


def plot_confusion_matrix(matrix, class_names, path):
    """Row-normalised heatmap. With ten classes the raw counts still fit in the cells."""
    normalised = matrix.float() / matrix.sum(dim=1, keepdim=True).clamp(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(normalised, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(image, ax=ax, fraction=0.046, label="fraction of the true class")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            count = matrix[i, j].item()
            if count == 0:
                continue
            ax.text(j, i, str(count), ha="center", va="center", fontsize=6,
                    color="black" if normalised[i, j] > 0.5 else "white")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set(xlabel="predicted class", ylabel="true class",
           title="Confusion matrix (row-normalised, counts in cells)")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("written:", path)


def plot_misclassified(images, labels, preds, class_names, mean, std, path, n=64):
    """Show n randomly sampled errors in an 8x8 grid, titled 'truth -> prediction'."""
    if len(images) == 0:
        print("no misclassified images - skipping plot")
        return

    generator = torch.Generator().manual_seed(SEED)
    sample = torch.randperm(len(images), generator=generator)[:n]
    shown = denormalize(images[sample], mean, std).squeeze(1).numpy()

    fig, axes = plt.subplots(8, 8, figsize=(10, 11))
    for ax in axes.flat:
        ax.axis("off")

    for ax, idx, image in zip(axes.flat, sample.tolist(), shown):
        ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"{class_names[labels[idx]]} -> {class_names[preds[idx]]}",
                     fontsize=8, color="firebrick")

    fig.suptitle(f"{len(shown)} of {len(images)} misclassified test images", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("written:", path)


# ----------------------------------------------------------------------
# Main functionality, utilazing the helper functions
# ----------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(SEED)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("device:", device)

    # --- normalization statistics, computed on training data only ---
    raw = datasets.MNIST(DATA_DIR, train=True, download=True,
                         transform=transforms.ToTensor())
    mean, std = channel_stats(raw, CHANNELS)
    print("mean:", [round(v, 4) for v in mean.tolist()])
    print("std: ", [round(v, 4) for v in std.tolist()])

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),
    ])

    train_ds = datasets.MNIST(DATA_DIR, train=True, transform=transform)
    test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

    persistent = NUM_WORKERS > 0
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, persistent_workers=persistent)
    test_dl = DataLoader(test_ds, batch_size=1000,
                         num_workers=NUM_WORKERS, persistent_workers=persistent)

    class_names = [str(digit) for digit in range(10)]
    print(f"train {len(train_ds)} | test {len(test_ds)}")

    # --- model, loss, optimizer ---
    model = build_model(len(class_names)).to(device)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    # --- training loop ---
    # No validation split here, same as in mnist_mlp.py: nothing is tuned on
    # MNIST, so the test set doubles as the per-epoch monitor. The CIFAR-100
    # run uses a proper three-way split.
    history = []
    batch_losses = []
    best_test_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, epoch_losses = train_one_epoch(
            model, train_dl, loss_fn, optimizer, device)
        test_loss, test_acc = evaluate(model, test_dl, loss_fn, device)

        batch_losses.extend(epoch_losses)
        history.append({"train_loss": train_loss, "train_acc": train_acc,
                        "test_loss": test_loss, "test_acc": test_acc})

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), f"{OUT_DIR}/{PREFIX}_best.pt")

        print(f"epoch {epoch:3d}/{EPOCHS} | "
              f"train loss {train_loss:.3f} acc {train_acc:6.2%} | "
              f"test loss {test_loss:.3f} acc {test_acc:6.2%}")

    print(f"\nbest test accuracy: {best_test_acc:.2%}")

    # --- error analysis ---
    model.load_state_dict(torch.load(f"{OUT_DIR}/{PREFIX}_best.pt", map_location=device))
    matrix = confusion_matrix(model, test_dl, device, len(class_names))
    print_top_confusions(matrix, class_names)

    # --- plots ---
    plot_loss(history, batch_losses, len(train_dl), f"{OUT_DIR}/{PREFIX}_loss.png")
    plot_confusion_matrix(matrix, class_names, f"{OUT_DIR}/{PREFIX}_confusion_matrix.png")

    mis_images, mis_labels, mis_preds = collect_misclassified(model, test_dl, device)
    plot_misclassified(mis_images, mis_labels, mis_preds, class_names, mean, std,
                       f"{OUT_DIR}/{PREFIX}_misclassified.png")