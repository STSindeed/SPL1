# NeuralForge

A feedforward neural network engine written in plain Java, no external
libraries. Built for Software Project Lab 1 (SPL-1), IIT, University of Dhaka.

Everything is written from scratch: matrices, forward pass, backpropagation,
optimizers, training loop, data loading, evaluation, model saving, and a Swing
window where you draw a digit and the network tells you which digit it is.

## What it can do

- **Matrix class** with the operations a network needs (multiply, transpose,
  element-wise operations, adding a bias row, row selection)
- **Dense layers** with ReLU, Leaky ReLU, Sigmoid, Tanh, Softmax and Linear
  activations, and Xavier and He weight initialisation
- **Loss functions:** mean squared error and cross-entropy
- **Backpropagation** with mini-batches, checked against numerical gradients
  (gradient checking)
- **Optimizers:** SGD, Momentum, Nesterov momentum, RMSProp and Adam. The ones with
  memory keep their own numbers for every weight between two updates.
- **Learning rate schedules:** constant, step, exponential, inverse-time and cosine
  decay
- **Regularization against overfitting:** L2 weight decay, dropout on the hidden
  layers (inverted dropout, so nothing changes at prediction time) and early
  stopping that goes back to the weights of the best epoch
- **Optimizer comparison:** trains the same network from the same starting weights
  with every optimizer and prints the results side by side
- **Regularization comparison:** trains the same network with L2, dropout and early
  stopping, alone and together, and shows the train/test gap and the validation loss
- **Data pipeline:** CSV reader (quoted fields, missing values), label encoder,
  stratified train/test split, standard and min-max scalers, IDX reader for MNIST
  (plain or `.gz`)
- **Training loop** that reports loss, accuracy and learning rate after every epoch
  on the training and test data, and saves the history as a CSV file
- **Metrics:** accuracy, confusion matrix, precision, recall, F1, top-k accuracy,
  most common mistakes
- **Model save and load** in its own binary format with a checksum
- **Digit recognizer window:** draw a digit with the mouse, press Predict

## Requirements

- JDK 11 or newer (checked with JDK 21)
- Nothing else. The window uses Swing, which is part of the JDK.

## Build and run

Linux / macOS / Git Bash:

    ./build.sh                    # compile and run the quick demos
    ./build.sh run iris           # compile and run one demo (see the table below)
    ./build.sh run mnist 10000 3  # arguments after "run" go to the program
    ./build.sh test               # compile and run all tests
    ./build.sh clean              # delete the out/ folder

Windows (cmd):

    build.bat
    build.bat run iris
    build.bat run mnist 10000 3
    build.bat test
    build.bat clean

Run the scripts from the project folder, because the demos look for `data/` and
`models/` relative to it. Loading all of MNIST needs about 2 GB of memory, the
scripts ask Java for that. If you get an `OutOfMemoryError`, set `JAVA_OPTS`
to a bigger value, for example `JAVA_OPTS=-Xmx4g ./build.sh run mnist`
(on Windows: `set JAVA_OPTS=-Xmx4g` first).

### The demos

| Command                                   | What it does                                                         | Needs              |
|-------------------------------------------|----------------------------------------------------------------------|--------------------|
| `run`                                     | runs `check`, `xor`, `blobs` and `iris` one after the other          | `data/iris.csv`    |
| `run check`                               | compares backpropagation gradients with numerical gradients         |                    |
| `run xor`                                 | learns XOR, which a single layer cannot do                           |                    |
| `run blobs`                               | classifies three groups of random points                             |                    |
| `run iris`                                | whole pipeline on a CSV file: load, split, scale, train, evaluate    | `data/iris.csv`    |
| `run mnist [n [e [optimizer [schedule [l2 [dropout [patience]]]]]]]` | trains on `n` images (0 = all 60000) for `e` epochs | MNIST files |
| `run optimizers [iris or mnist [n [e]]]`  | trains with every optimizer and prints a comparison table            | iris or MNIST      |
| `run decay [n [e [optimizer [rate]]]]`    | trains with every learning rate schedule and prints a table          | MNIST files        |
| `run regularize [n [e [l2 [dropout [patience]]]]]` | trains with L2, dropout and early stopping and prints a comparison table | MNIST files |
| `run eval [file]`                         | loads a saved model and measures it on the 10000 test images         | MNIST files, model |
| `run gui [file]`                          | opens the window where you draw a digit                              | model              |

Optimizers: `sgd`, `momentum`, `nesterov`, `rmsprop`, `adam`.
Schedules: `constant`, `step`, `exp`, `inverse`, `cosine`.

The last three arguments of `mnist` switch on regularization: `l2` is the strength of
the L2 penalty (for example 0.001), `dropout` the share of hidden neurons that is
switched off in every step (for example 0.3) and `patience` the number of epochs without
improvement before early stopping ends the training. With `patience` above 0, 10% of the
training images are held back for early stopping to watch, so the test images stay
untouched and the final test accuracy is honest.

`mnist` without arguments trains on all 60000 images for 5 epochs with `sgd` and a
constant rate of 0.1. The other optimizers start with their usual rates (0.02 for
momentum and nesterov, 0.001 for rmsprop and adam). The trained model is saved to
`models/mnist.nfm` and the loss curve, with the learning rate of every epoch, to
`results/mnist_history.csv`.

`optimizers` uses `n` = 10000 images and 5 epochs on MNIST by default, and 60
epochs on Iris. `decay` uses 10000 images and 10 epochs, with Adam at a start rate
of 0.01, which is a bit too high on purpose (see Results).

**Careful:** `mnist` always overwrites `models/mnist.nfm`. A short test run such
as `run mnist 1000 1` replaces a good model with a weak one. Copy the model
somewhere else first if you want to keep it.

### A typical session

    ./build.sh run mnist 0 10 adam cosine   # train on all images, save the model
    ./build.sh run eval                     # check the saved model: accuracy, confusion matrix
    ./build.sh run gui                      # draw digits (left mouse button), Clear, Predict

## Data

**Iris** (`data/iris.csv`) is part of the repository.

**MNIST** is not, the files are too big. Download the four files and put them
in `data/mnist/`, with exactly these names (hyphens, not dots):

    train-images-idx3-ubyte     60000 training images
    train-labels-idx1-ubyte
    t10k-images-idx3-ubyte      10000 test images
    t10k-labels-idx1-ubyte

They must be files, not folders. The `.gz` versions work too, the reader looks at
the content of the file and not at its name. See `data/mnist/README.txt`.

## Tests

    ./build.sh test

runs all 30 test classes. They use a small test runner of our own
(`TestHelper`), no JUnit. Every class prints how many checks passed and failed.
The script runs all classes even if one fails, lists the failed ones at the end
and exits with an error.

The optimizer tests check every update rule against numbers worked out by hand,
and check that every optimizer really learns a simple function.

The regularization tests check the dropout mask (values, share of dropped neurons,
average output), the dropout gradient against a numerical gradient with a fixed mask,
the L2 penalty and its gradient (also through the gradient checker), and the early
stopping rule with made-up loss curves, including restoring the best weights.

## Folder layout

    build.sh, build.bat                     build and run scripts
    data/iris.csv                           the Iris dataset
    data/mnist/                             put the MNIST files here (not committed)
    models/mnist.nfm                        trained model used by eval and gui
    results/                                loss curves written by the demos (not committed)
    presentations/                          proposal and presentation files
    resources/                              learning material (videos.txt)

    src/main/java/neuralforge/              main source code
        Main.java                           the demos
        math/                               Matrix class
        activation/                         activation functions
        init/                               weight initialisers
        layer/                              dense layer, with optional dropout
        loss/                               loss functions
        optimizer/                          SGD, Momentum, RMSProp, Adam and a factory
        schedule/                           learning rate schedules
        network/                            network and mini-batch training
        gradcheck/                          gradient checking
        data/                               CSV and IDX readers, Dataset, label encoder, split
        preprocess/                         standard and min-max scalers
        training/                           training loop, evaluation, history, early stopping
        experiment/                         comparison of optimizers, schedules and regularization
        metrics/                            confusion matrix, top-k accuracy
        io/                                 saving and loading models
        gui/                                the digit recognizer window
    src/test/java/neuralforge/              tests, one test class per main class
    out/                                    compiled classes (not committed)

## How the parts fit together

    CSV or IDX file
      -> DatasetLoader -> Dataset
      -> stratified split, then scaler (fitted on the training part only)
      -> Trainer: NeuralNetwork + loss + optimizer, mini-batches, every epoch measured
      -> Evaluation -> ConfusionMatrix, top-k accuracy
      -> ModelIO.save -> models/mnist.nfm -> ModelIO.load -> DigitApp (the window)

While the trainer runs the mini-batches, the network is in training mode and dropout
is active. Before every measurement on the validation or test data the trainer switches
back to prediction mode, so measurements never depend on chance. After every epoch the
early stopping rule looks at the validation loss, and when it gives up, the weights of the
best epoch are put back.

At the start of every epoch the trainer tells the optimizer which epoch begins, so
an optimizer with a learning rate schedule can switch to the rate of that epoch.

The window draws directly on a 28 x 28 grid, the same size as an MNIST image, so
what is shown on the screen is exactly what the network gets as input.

## Results

The random seeds are fixed in `Main`, so a run should give the same numbers again.

**Final models** (MNIST, 784-128-10, batch size 64, test accuracy):

| Optimizer and schedule              | Epochs | Test accuracy                  |
|-------------------------------------|--------|--------------------------------|
| sgd 0.1, constant                   | 5      | 96.76%                         |
| sgd 0.1, constant                   | 10     | 97.58%                         |
| adam 0.001, cosine decay            | 10     | 97.71% (the saved model)       |

Iris (4-8-3, SGD 0.1, 120 training and 30 test samples): 93.3%, that is 28 of 30.
The test set is small, so one sample changes the result by 3.3 points.

**Optimizers** (`run optimizers mnist`: 10000 training images, 5 epochs, every
optimizer with its usual start rate, same starting weights and same shuffling):

| Optimizer | Test accuracy | Epochs to reach 92% |
|-----------|---------------|---------------------|
| sgd       | 92.21%        | 5                   |
| momentum  | 93.44%        | 3                   |
| nesterov  | 93.27%        | 3                   |
| rmsprop   | 93.78%        | 3                   |
| adam      | 93.69%        | 3                   |

All the others are ahead of plain SGD. On Iris (60 epochs, only 30 test samples)
RMSProp and Adam with their small usual rate of 0.001 are behind SGD and momentum,
they need a bigger rate or more epochs on such a tiny problem.

**Learning rate decay** (`run decay`: 10000 training images, 10 epochs):

| Schedule  | Adam, start rate 0.01 (too high) | Nesterov, start rate 0.02 (well chosen) |
|-----------|----------------------------------|------------------------------------------|
| constant  | 95.01%                           | 94.17%                                   |
| step      | 95.92%                           | 93.11%                                   |
| exp       | 95.90%                           | 93.04%                                   |
| inverse   | 95.82%                           | 92.80%                                   |
| cosine    | 96.05%                           | 93.77%                                   |

When the rate is too high, a constant rate keeps the weights jumping around and
every decay schedule is better. When the rate is already well chosen for a short
training, decay gives nothing and even costs a little, because the rate gets small
before the network has finished learning. These are single runs, so differences of
a few tenths of a percent should not be over-read.

(The second column was made with `run decay 10000 10 nesterov 0.02`, the first with
the defaults of `run decay`.)

**Regularization** (`run regularize`: 1698 training, 302 validation and 10000 test images,
784-256-10 network, Adam 0.001, up to 30 epochs, L2 0.003, dropout 0.4, patience 6):

| Setup        | Epochs run | Train acc | Test acc | Gap    | Test loss |
|--------------|------------|-----------|----------|--------|-----------|
| none         | 30         | 100.00%   | 91.16%   | 8.84%  | 0.3317    |
| l2           | 30         | 99.94%    | 90.84%   | 9.10%  | 0.3028    |
| dropout      | 30         | 100.00%   | 91.73%   | 8.27%  | 0.2993    |
| early        | 13         | 98.23%    | 90.32%   | 7.91%  | 0.3277    |
| l2+dropout   | 30         | 99.65%    | 90.90%   | 8.75%  | 0.2971    |
| all three    | 18         | 98.35%    | 90.44%   | 7.91%  | 0.3253    |

Without regularization the validation loss is lowest at epoch 9 (0.2907) and then rises to
0.3289 at epoch 30 while the training accuracy is already 100%, which is overfitting. L2
and dropout keep the validation loss from rising and lower the test loss by about
10%. The test accuracy changes by less than a point, which is within what a single run
can vary, so it should not be over-read. Early stopping cuts the training time by more than
half, but it watches the loss, and the accuracy was still creeping up slightly after the
loss turned, so it ends a little lower here. These are single runs on a small training set.

## Known limits

- Digits that are drawn small, off-center or in an unusual style are not always
  recognized. MNIST digits are centered by their center of mass, and the window
  does not do that yet.
- Only dense layers. Training runs on one CPU thread.
- The optimizer state is not saved with the model, so training cannot be resumed
  exactly where it stopped. L2, dropout and early stopping are training settings only,
  the model file holds weights and biases, nothing else.
- L2 is added to the gradient (classic L2), which for Adam is not the same as the
  decoupled weight decay of AdamW.

## Planned

More robust data handling
(categorical columns, automatic task detection), a command line with `train`,
`predict` and `evaluate`, hyperparameter search with cross-validation, faster
matrix multiplication, a training report, and tests on more datasets.
