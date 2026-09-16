# Image classification from scratch: MNIST and CIFAR-100

Three PyTorch training scripts, written to learn how convolutional networks
behave rather than to chase a benchmark. Every model is trained from scratch on
a MacBook (M2 Pro, MPS), no pretrained weights anywhere.

The CIFAR-100 part is the interesting one: four controlled runs that isolate
what longer training, augmentation and model capacity each contribute.

## Results

| Script | Dataset | Parameters | Test accuracy |
|---|---|---|---|
| `src/mnist_mlp.py` | MNIST | 101,770 | 97.77 % |
| `src/mnist_cnn.py` | MNIST | 421,642 | 99.16 % |
| `src/cifar100_cnn.py` | CIFAR-100 | 4,738,596 | 76.20 % |

The two MNIST scripts use the same optimizer, learning rate, batch size and
epoch count, so the only difference is the model. Two convolution blocks instead
of one hidden layer cut the errors from 223 to 84 out of 10,000 test images, for
4.1 times the parameters. In accuracy that reads as a modest 1.39 points, which
is why error counts are the more honest number this far up the scale.

The CIFAR-100 run takes about 52 minutes on an M2 Pro, roughly 21 seconds per
epoch.

## The CIFAR-100 ablation

Each row changes one thing against the row above it.

| Run | Epochs | Augmentation | Parameters | Test accuracy | Δ |
|---|---|---|---|---|---|
| A | 40 | crop + flip | 1.17 M | 70.44 % | baseline |
| B | 150 | + TrivialAugmentWide + RandomErasing | 1.17 M | 72.50 % | +2.06 |
| C | 150 | + TrivialAugmentWide + RandomErasing | 4.74 M | 76.20 % | +3.70 |
| D | 250 | + TrivialAugmentWide + RandomErasing | 4.74 M | 76.43 % | +0.23 |

Two things came out of this that I would not have guessed.

Capacity mattered more than regularisation. Going from three conv blocks to four
(B to C) bought 3.70 points, while stronger augmentation plus almost four times
the epochs (A to B) bought 2.06. I had expected the opposite order, because run
A overfitted badly: train 84.3 % against val 70.8 %. Run B shows why capacity was
the binding constraint. Its training accuracy ended 9 points *below* validation,
so that model could not even fit the augmented training data.

Training time saturates. Run D spent 100 extra epochs for 0.23 points, and the
train-validation gap reopened from +0.25 to +2.97. Past 150 epochs the model only
learns the training set better. The script therefore defaults to 150.

Run C was repeated after the code was restructured and reproduced epoch by epoch,
including the final 76.20 %.

Caveat: one seed per configuration. I never measured how much two runs of the
same configuration differ, so the +0.23 in run D is probably noise, and the
larger deltas are likely but not provably real.

## What the errors look like

Every run prints its most frequent confusions and writes a confusion matrix to
`results/`.

On CIFAR-100 almost every remaining error sits inside one of the superclasses:

- trees: maple, oak, pine and willow, confused in both directions
- people: boy, girl, man, woman, baby
- aquatic animals: otter and seal, dolphin and shark

Symmetric confusions like these are what you would expect from classes that are
hard to tell apart at 32×32, rather than from a model that learned something
wrong. Worst class by recall is `otter` at 49 %.

MNIST shows the same structure at a much smaller scale. The MLP's single worst
confusion is 7 predicted as 9, 21 times. The CNN's worst is 3 predicted as 5,
7 times. The pair 4 and 9 survives in both models.

## Setup

```bash
uv sync
cd src
uv run python cifar100_cnn.py
```

Datasets download themselves into `src/data/` on first run. Figures and
checkpoints are written to `results/`.

Set `NUM_WORKERS = 0` in the script when running inside Jupyter. On macOS the
DataLoader workers re-import the file, which is why everything that executes sits
below `if __name__ == "__main__"`.

## Method notes

Normalisation statistics (per channel mean and std) are computed on the training
split only.

CIFAR-100 uses a 45,000 / 5,000 / 10,000 train-validation-test split. The test
set is evaluated exactly once, at the end, with the checkpoint that had the best
validation accuracy. The MNIST scripts have no validation split: nothing is tuned
on MNIST, so the test set doubles as the per-epoch monitor. That is a shortcut,
and it would not be defensible if any hyperparameter had been chosen from those
numbers.

Augmentation is applied to the training split only. Both splits point at the same
raw files through two dataset objects with different transforms.

`AdaptiveAvgPool2d(1)` instead of a large `Flatten` into a dense layer. In the
first version of the CIFAR model the first linear layer held 8.39 M of 8.72 M
parameters. Global average pooling removed almost all of that and made the model
independent of the input size.

## What this is not

A benchmark result. Published from-scratch numbers on CIFAR-100 reach about 85 %,
and anything above 90 % in the literature involves ImageNet pretraining. 76 % is
what a small VGG-style network gets on a laptop.

## Files

```
├── README.md
├── pyproject.toml
├── uv.lock
├── src/
│   ├── mnist_mlp.py
│   ├── mnist_cnn.py
│   └── cifar100_cnn.py
└── results/
    ├── mnist_mlp_*.png
    ├── mnist_cnn_*.png
    └── cifar100_*.png
```