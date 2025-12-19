# Deep Learning Optimizer Comparison: MNIST Classification

This project provides a systematic evaluation of various optimization algorithms on the MNIST dataset using a custom Convolutional Neural Network (CNN). By conducting a grid search across 21 different hyperparameter configurations, this study explores the performance trade-offs between classical SGD and modern adaptive optimizers like Adam and RMSProp.

## Model Architecture

The model is a 4-layer architecture designed for efficient feature extraction:

- **Convolutional Block 1:** 32 filters (3x3), ReLU activation, 2x2 Max-Pooling.
- **Convolutional Block 2:** 64 filters (3x3), ReLU activation, 2x2 Max-Pooling.
- **Fully-Connected Layer:** 128 hidden units with ReLU activation.
- **Output Layer:** 10 units for digit classification.

## Experimental Setup

A hyperparameter grid search was conducted to evaluate four major optimizers:

- **SGD:** Tested across learning rates {0.1, 0.01, 0.001} and momentum values {0.0, 0.5, 0.9}.
- **RMSProp:** Learning rates from 0.01 to 0.0005.
- **AdaGrad:** Learning rates from 0.1 to 0.005.
- **Adam:** Learning rates from 0.01 to 0.0005.

### Training Details:

- **Dataset:** Standard MNIST (60,000 training / 10,000 test).
- **Normalization:** Zero mean ($\mu=0.1307$) and unit variance ($\sigma=0.3081$).
- **Epochs:** 15.
- **Hardware:** Accelerated via CUDA on GPU.

## Key Findings

The most successful configuration was Stochastic Gradient Descent (SGD) with a learning rate of 0.1 and momentum of 0.5, achieving a final validation accuracy of 99.38%.

| Optimizer | Best Configuration | Val Accuracy |
|-----------|-------------------|--------------|
| SGD | $lr=0.1$, $mom=0.5$ | 99.38% |
| RMSProp | $lr=0.0005$ | 99.17% |
| Adam | $lr=0.001$ | 99.09% |
| AdaGrad | $lr=0.05$ | 98.85% |

### Why did SGD win?

While adaptive optimizers like Adam and RMSProp are often default choices, they underperformed plain SGD on this specific task.

- **Uniform Gradients:** MNIST features relatively uniform gradient scales, meaning the per-parameter adaptation of Adam provides minimal benefit.
- **Convergence Speed:** A higher learning rate (0.1) in SGD allowed for faster convergence, reaching >99% accuracy within 5 epochs.
- **Stability:** Moderate momentum (0.5) provided trajectory smoothing without the instability seen in higher momentum values (0.9).
