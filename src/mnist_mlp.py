"""MNIST: fully connected baseline (784 - 128 - 10).

The simplest model in this repo, kept as the reference point the CNNs are
measured against. Architecture, optimizer and epoch count are unchanged from
the original version; only the structure of the script and the evaluation
were reworked.

Writes best.pt and three PNGs to ../results/.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from helper import plot_confusion_matrix, plot_loss, plot_misclassified

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
PREFIX = "mnist_mlp"


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
    """Flatten the image and push it through one hidden layer."""
    return nn.Sequential(
        nn.Flatten(),           # (B,1,28,28) -> (B,784)
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes),   # logits, no softmax
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
    # No validation split here: MNIST is the baseline and nothing is tuned on
    # it, so the test set doubles as the per-epoch monitor. The CIFAR-100 run
    # uses a proper three-way split.
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
                       f"{OUT_DIR}/{PREFIX}_misclassified.png", seed=SEED)