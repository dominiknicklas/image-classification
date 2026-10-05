"""MNIST: the MLP from mnist_mlp.py again, this time without any framework.

Same architecture (784 - 128 - 10, ReLU), same loss, optimizer, learning rate,
batch size and epoch count as the PyTorch version. Forward pass, backward pass
and the SGD update all run on the Value class from micrograd.py, so the only
thing imported for the computation is Python's math module. The MNIST files
are parsed by hand as well.

It is slow: 53 minutes for the five epochs on an M2 Pro, against seconds in
PyTorch. Reaches 97.90 % test accuracy.

Writes best.json and three PNGs to ../results/.
"""

import gzip
import json
import math
import os
import random
import struct
import time
import urllib.request

from helper import plot_confusion_matrix, plot_loss, plot_misclassified
from micrograd import MLP, SGD, cross_entropy

# ----------------------------------------------------------------------
# Hyperparameters - the only place to tune
# ----------------------------------------------------------------------
EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.1
HIDDEN = 128
TRAIN_SIZE = None        # None = all 60,000; a small number makes a quick test run
TEST_SIZE = None         # None = all 10,000
LOG_EVERY = 100          # print a progress line every this many batches
SEED = 0
DATA_DIR = "data"
OUT_DIR = "../results"
PREFIX = "mnist_mlp_micrograd"
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist/"
IMAGE_SIZE = 28


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def read_idx(name):
    """Return the raw bytes of an MNIST IDX file, downloading it if it is missing."""
    path = f"{DATA_DIR}/MNIST/raw/{name}"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()

    if not os.path.exists(path + ".gz"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print("downloading", name)
        urllib.request.urlretrieve(MNIST_URL + name + ".gz", path + ".gz")
    with gzip.open(path + ".gz", "rb") as f:
        return f.read()


def load_images(name, limit=None):
    """Parse an IDX image file into one bytes object of 784 grey values per image."""
    data = read_idx(name)
    # Header: magic number, image count, rows, columns - four big-endian ints.
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    assert magic == 2051, "not an IDX image file"
    size = rows * cols
    count = min(count, limit or count)
    return [data[16 + i * size:16 + (i + 1) * size] for i in range(count)]


def load_labels(name, limit=None):
    """Parse an IDX label file into a list of ints."""
    data = read_idx(name)
    magic, count = struct.unpack(">II", data[:8])
    assert magic == 2049, "not an IDX label file"
    return list(data[8:8 + min(count, limit or count)])


def pixel_stats(images):
    """Mean and standard deviation of all pixels, scaled to [0, 1]."""
    # 47 million pixels, but only 256 different values: count each value once.
    joined = b"".join(images)
    counts = [joined.count(value) for value in range(256)]
    n = len(joined)

    mean = sum(count * value / 255 for value, count in enumerate(counts)) / n
    mean_sq = sum(count * (value / 255) ** 2 for value, count in enumerate(counts)) / n
    return mean, (mean_sq - mean**2) ** 0.5


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def train_one_epoch(model, images, labels, table, optimizer, rng, epoch):
    """Run one pass over the training data. Returns loss, accuracy, per-batch losses."""
    order = list(range(len(images)))
    rng.shuffle(order)
    steps = (len(order) + BATCH_SIZE - 1) // BATCH_SIZE

    loss_sum, correct = 0.0, 0
    batch_losses = []
    start = time.time()

    for step in range(steps):
        batch = order[step * BATCH_SIZE:(step + 1) * BATCH_SIZE]

        # Forward: one graph per image, all of them joined by the mean loss.
        losses = []
        for index in batch:
            pixels = [table[byte] for byte in images[index]]
            logits = model(pixels)
            losses.append(cross_entropy(logits, labels[index]))
            correct += argmax([logit.data for logit in logits]) == labels[index]
        loss = sum(losses[1:], losses[0]) * (1 / len(batch))

        # Backward and update: gradients of the batch mean, one SGD step.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_losses.append(loss.data)
        loss_sum += loss.data * len(batch)

        if (step + 1) % LOG_EVERY == 0:
            print(f"  epoch {epoch} | batch {step + 1:4d}/{steps} | "
                  f"loss {loss.data:.3f} | {time.time() - start:5.0f} s", flush=True)

    return loss_sum / len(order), correct / len(order), batch_losses


def evaluate(model, images, labels, table):
    """Return average loss, accuracy and the predictions over a dataset."""
    loss_sum, correct = 0.0, 0
    preds = []

    for image, label in zip(images, labels):
        logits = model.predict([table[byte] for byte in image])
        preds.append(argmax(logits))
        correct += preds[-1] == label

        # Cross-entropy on plain floats, same shift as in micrograd.cross_entropy.
        shift = max(logits)
        log_sum = math.log(sum(math.exp(logit - shift) for logit in logits))
        loss_sum += log_sum - (logits[label] - shift)

    return loss_sum / len(images), correct / len(images), preds


def confusion_matrix(labels, preds, num_classes):
    """Count predictions per (true class, predicted class) pair.

    Row i, column j holds how often an image of class i was predicted as
    class j. The diagonal are the correct ones, everything else is an error.
    """
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for label, pred in zip(labels, preds):
        matrix[label][pred] += 1
    return matrix


def print_top_confusions(matrix, class_names, n=10):
    """List the class pairs the model mixes up most often."""
    errors = [(count, true_idx, pred_idx)
              for true_idx, row in enumerate(matrix)
              for pred_idx, count in enumerate(row)
              if true_idx != pred_idx]  # correct predictions are not confusions
    errors.sort(reverse=True)

    print(f"\ntop {n} confusions (count | true -> predicted):")
    for count, true_idx, pred_idx in errors[:n]:
        print(f"  {count:4d} | {class_names[true_idx]:>10} -> {class_names[pred_idx]}")


def save_weights(model, path):
    with open(path, "w") as f:
        json.dump([p.data for p in model.parameters()], f)


def load_weights(model, path):
    with open(path) as f:
        for p, value in zip(model.parameters(), json.load(f)):
            p.data = value


# ----------------------------------------------------------------------
# Main functionality, utilazing the helper functions
# ----------------------------------------------------------------------
def main():
    random.seed(SEED)

    # --- data ---
    train_images = load_images("train-images-idx3-ubyte", TRAIN_SIZE)
    train_labels = load_labels("train-labels-idx1-ubyte", TRAIN_SIZE)
    test_images = load_images("t10k-images-idx3-ubyte", TEST_SIZE)
    test_labels = load_labels("t10k-labels-idx1-ubyte", TEST_SIZE)

    # --- normalization statistics, computed on training data only ---
    mean, std = pixel_stats(train_images)
    print("mean:", [round(mean, 4)])
    print("std: ", [round(std, 4)])

    # A byte can only take 256 values, so normalising is a table lookup.
    table = [(value / 255 - mean) / std for value in range(256)]

    class_names = [str(digit) for digit in range(10)]
    print(f"train {len(train_images)} | test {len(test_images)}")

    # --- model, optimizer ---
    model = MLP(IMAGE_SIZE * IMAGE_SIZE, [HIDDEN, len(class_names)])
    print(f"parameters: {len(model.parameters()):,}")

    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
    shuffle_rng = random.Random(SEED)

    # --- training loop ---
    # No validation split here, same as in mnist_mlp.py: nothing is tuned on
    # MNIST, so the test set doubles as the per-epoch monitor.
    history = []
    batch_losses = []
    best_test_acc = 0.0
    checkpoint = f"{OUT_DIR}/{PREFIX}_best.json"
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, epoch_losses = train_one_epoch(
            model, train_images, train_labels, table, optimizer, shuffle_rng, epoch)
        test_loss, test_acc, _ = evaluate(model, test_images, test_labels, table)

        batch_losses.extend(epoch_losses)
        history.append({"train_loss": train_loss, "train_acc": train_acc,
                        "test_loss": test_loss, "test_acc": test_acc})

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_weights(model, checkpoint)

        print(f"epoch {epoch:3d}/{EPOCHS} | "
              f"train loss {train_loss:.3f} acc {train_acc:6.2%} | "
              f"test loss {test_loss:.3f} acc {test_acc:6.2%} | "
              f"{(time.time() - start) / 60:.0f} min", flush=True)

    print(f"\nbest test accuracy: {best_test_acc:.2%}")

    # --- error analysis ---
    load_weights(model, checkpoint)
    _, _, preds = evaluate(model, test_images, test_labels, table)
    matrix = confusion_matrix(test_labels, preds, len(class_names))
    print_top_confusions(matrix, class_names)

    # --- plots ---
    steps_per_epoch = len(batch_losses) // EPOCHS
    plot_loss(history, batch_losses, steps_per_epoch, f"{OUT_DIR}/{PREFIX}_loss.png")
    plot_confusion_matrix(matrix, class_names, f"{OUT_DIR}/{PREFIX}_confusion_matrix.png")

    # Normalised (1,28,28) images, the layout plot_misclassified expects.
    wrong = [i for i, (label, pred) in enumerate(zip(test_labels, preds)) if label != pred]
    mis_images = [[[[table[byte] for byte in test_images[i][row * IMAGE_SIZE:(row + 1) * IMAGE_SIZE]]
                    for row in range(IMAGE_SIZE)]]
                  for i in wrong]
    plot_misclassified(mis_images, [test_labels[i] for i in wrong], [preds[i] for i in wrong],
                       class_names, [mean], [std],
                       f"{OUT_DIR}/{PREFIX}_misclassified.png", seed=SEED)


if __name__ == "__main__":
    main()
