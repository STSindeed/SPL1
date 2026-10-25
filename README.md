# NeuralForge

A feedforward neural network engine written in plain Java, no external
libraries. Built for Software Project Lab 1 (SPL-1), IIT, University of Dhaka.

The network is trained from scratch (matrices, forward pass, backpropagation,
optimizers) and will later be used for digit recognition through a Swing GUI,
and for other datasets too.

## Requirements

- JDK 11 or newer (checked with JDK 21)

## Build and run

Linux / macOS / Git Bash:

    ./build.sh            # compile and run the demo
    ./build.sh test       # compile and run all tests
    ./build.sh clean      # delete the out/ folder

Windows (cmd):

    build.bat
    build.bat test
    build.bat clean

## Folder layout

    src/main/java/neuralforge/              main source code
    src/main/java/neuralforge/math/         Matrix class
    src/main/java/neuralforge/activation/   activation functions
    src/main/java/neuralforge/init/         weight initialisers
    src/main/java/neuralforge/layer/        dense layer
    src/main/java/neuralforge/loss/         loss functions
    src/main/java/neuralforge/optimizer/    optimizers (SGD)
    src/main/java/neuralforge/network/      network and training loop
    src/main/java/neuralforge/gradcheck/    gradient checking
    src/test/java/neuralforge/              tests (own small runner, no JUnit)
    out/                                    compiled classes (not committed)
