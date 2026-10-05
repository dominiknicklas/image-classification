"""CIFAR-100: VGG-style CNN with BatchNorm, augmentation and a OneCycle LR schedule.

Reaches 76.2 % test accuracy with the settings below, trained from scratch on
an M2 Pro. See the README for the ablation the configuration came out of.

Writes best.pt and three PNGs to ../results/.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from helper import plot_confusion_matrix, plot_loss, plot_misclassified

# ----------------------------------------------------------------------
# Hyperparameters - the only place to tune
# ----------------------------------------------------------------------
EPOCHS = 150             # 250 was measured and gained 0.23 points, see README
BATCH_SIZE = 128
VAL_SIZE = 5_000
MAX_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
LABEL_SMOOTHING = 0.1
DROPOUT = 0.3
CHANNELS = 3
NUM_WORKERS = 4          # set to 0 when running inside Jupyter
SEED = 0
DATA_DIR = "data"
OUT_DIR = "../results"
PREFIX = "cifar100"
TOP_CONFUSIONS = 20      # how many confused class pairs to print


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


def conv_block(c_in, c_out, pool=True):
    """Two 3x3 convolutions with BatchNorm, optionally halving the spatial size."""
    layers = [
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def build_model(num_classes):
    """Four conv blocks, global average pooling, one linear classifier.

    The fourth block widens to 512 channels without pooling: at 4x4 there is
    no spatial resolution left to give away.
    """
    return nn.Sequential(
        conv_block(3, 64),                 # (B,3,32,32) -> (B,64,16,16)
        conv_block(64, 128),               #             -> (B,128,8,8)
        conv_block(128, 256),              #             -> (B,256,4,4)
        conv_block(256, 512, pool=False),  #             -> (B,512,4,4)
        nn.AdaptiveAvgPool2d(1),           #             -> (B,512,1,1)
        nn.Flatten(),                      #             -> (B,512)
        nn.Dropout(DROPOUT),
        nn.Linear(512, num_classes),
    )


def train_one_epoch(model, dataloader, loss_fn, optimizer, scheduler, device):
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
        scheduler.step()  # OneCycle advances per batch, not per epoch

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


def print_top_confusions(matrix, class_names, n=TOP_CONFUSIONS):
    """List the class pairs the model mixes up most often."""
    errors = matrix.clone()
    errors.fill_diagonal_(0)  # correct predictions are not confusions
    top = errors.flatten().topk(n)

    print(f"\ntop {n} confusions (count | true -> predicted):")
    for count, index in zip(top.values.tolist(), top.indices.tolist()):
        true_idx, pred_idx = divmod(index, len(class_names))
        print(f"  {count:4d} | {class_names[true_idx]:>16} -> {class_names[pred_idx]}")


def print_worst_classes(matrix, class_names, n=10):
    """List the classes with the lowest recall (hardest for the model)."""
    recall = matrix.diagonal().float() / matrix.sum(dim=1).clamp(min=1)
    worst = recall.topk(n, largest=False)

    print(f"\nworst {n} classes by recall:")
    for value, index in zip(worst.values.tolist(), worst.indices.tolist()):
        print(f"  {value:6.1%} | {class_names[index]}")


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
    raw = datasets.CIFAR100(DATA_DIR, train=True, download=True,
                            transform=transforms.ToTensor())
    mean, std = channel_stats(raw, CHANNELS)
    print("mean:", [round(v, 4) for v in mean.tolist()])
    print("std: ", [round(v, 4) for v in std.tolist()])

    # --- transforms: augmentation for training, deterministic for evaluation ---
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(),      # operates on PIL images
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),
        transforms.RandomErasing(p=0.25),     # needs a tensor, so it goes last
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean.tolist(), std.tolist()),
    ])

    # --- train/val split ---
    # Two dataset objects over the same raw files, so the training split gets
    # augmentation while the validation split stays deterministic.
    train_full = datasets.CIFAR100(DATA_DIR, train=True, transform=train_tf)
    val_full = datasets.CIFAR100(DATA_DIR, train=True, transform=eval_tf)
    test_ds = datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=eval_tf)

    generator = torch.Generator().manual_seed(SEED)
    perm = torch.randperm(len(train_full), generator=generator).tolist()
    train_ds = Subset(train_full, perm[:-VAL_SIZE])
    val_ds = Subset(val_full, perm[-VAL_SIZE:])

    persistent = NUM_WORKERS > 0
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                          num_workers=NUM_WORKERS, persistent_workers=persistent)
    val_dl = DataLoader(val_ds, batch_size=512,
                        num_workers=NUM_WORKERS, persistent_workers=persistent)
    test_dl = DataLoader(test_ds, batch_size=512,
                         num_workers=NUM_WORKERS, persistent_workers=persistent)

    class_names = test_ds.classes
    print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)}")

    # --- model, loss, optimizer, schedule ---
    model = build_model(len(class_names)).to(device)
    print(f"classes: {len(class_names)} | "
          f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.SGD(model.parameters(), lr=MAX_LR, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY, nesterov=True)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=len(train_dl))

    # --- training loop ---
    history = []
    batch_losses = []
    best_val_acc = 0.0
    checkpoint = f"{OUT_DIR}/{PREFIX}_best.pt"

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, epoch_losses = train_one_epoch(
            model, train_dl, loss_fn, optimizer, scheduler, device)
        val_loss, val_acc = evaluate(model, val_dl, loss_fn, device)

        batch_losses.extend(epoch_losses)
        history.append({"train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})

        # Keep the best epoch, not the last one.
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint)

        print(f"epoch {epoch:3d}/{EPOCHS} | "
              f"train loss {train_loss:.3f} acc {train_acc:6.2%} | "
              f"val loss {val_loss:.3f} acc {val_acc:6.2%} | "
              f"lr {scheduler.get_last_lr()[0]:.4f}")

    # --- final evaluation: touch the test set exactly once ---
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    test_loss, test_acc = evaluate(model, test_dl, loss_fn, device)
    print(f"\nbest val accuracy: {best_val_acc:.2%}")
    print(f"test accuracy:     {test_acc:.2%}")

    # --- error analysis ---
    matrix = confusion_matrix(model, test_dl, device, len(class_names))
    print_top_confusions(matrix, class_names)
    print_worst_classes(matrix, class_names)

    # --- plots ---
    plot_loss(history, batch_losses, len(train_dl), f"{OUT_DIR}/{PREFIX}_loss.png",
              eval_split="val")
    # A hundred classes: no counts in the cells, bigger canvas, smaller labels.
    plot_confusion_matrix(matrix, class_names, f"{OUT_DIR}/{PREFIX}_confusion_matrix.png",
                          show_counts=False, figsize=(14, 13), tick_fontsize=4, dpi=200)

    mis_images, mis_labels, mis_preds = collect_misclassified(model, test_dl, device)
    plot_misclassified(mis_images, mis_labels, mis_preds, class_names, mean, std,
                       f"{OUT_DIR}/{PREFIX}_misclassified.png", seed=SEED,
                       figsize=(12, 13), title="True: {true}\n- Pred: {pred}",
                       title_fontsize=7)