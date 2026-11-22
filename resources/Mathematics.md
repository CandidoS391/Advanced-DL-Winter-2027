# Mathematical Foundations

This document provides a review of the mathematical concepts commonly found in deep learning, with emphasis on neural networks and convolutional neural networks (CNNs).

The material presented here is only mean't as a refresher for participants with prior experience with the math involved. It should not be treated as a substitute for a full course or anything else equivalent in linear algebra, calculus, or probability/statistics.

## Linear Algebra
In terms of deep learning, and all other branches of artificial intelligence, linear algebra provides the mathematical framework for representing data and performing computations within models.

### 1.1 Vectors and Matrices
A vector is an ordered collection of values, or a one dimensional array. For instance:

```math
\mathbf{x} =
\begin{bmatrix}
x_1 \\
x_2 \\
\vdots \\
x_n
\end{bmatrix}
\in \mathbb{R}^n
```

A matrix meanwhile is a two dimensional array of values, written as

```math
W =
\begin{bmatrix}
  w_{11} & w_{12} \\
  w_{21} & w_{22}
\end{bmatrix}.
```

For neural networks, vectors can represent individual observations, activations, or biases. While matrices represent the weights connecting the fully connected layers.

### 1.2 Dot Products
For two vectors $`x, y \in \mathbf{R}^n`$, the dot product is defined as

```math
\sum_{i=1}^nx_iy_i.
```

Dot products appear in fully connected neural networks, where individual neurons compute weighted combinations of their inputs.

### 1.3 Matrix Multiplication
For compatible matrices $`A \in \mathbb{R}^{m \times n}, B \in \mathbb{R}^{n \times p},`$ Their product satisfies:

```math
AB \in \mathbb{R}^{m \times p}.
```

For each entry is computed by:

```math
\sum_{k=1}^{n}A_{ik}B_{kj}.
```

A fully connected neural network layer can be expressed as 

```math
W\mathbf{x} + \mathbf{b}.
```

Here, $W$ contains the trainable weights, $\mathbf{x}$ is the input, and $\mathbf{b}$ is the bias vector.

### 1.4 Tensors
Tensors generalize the multidimensional array representations used in machine learning. For instance, an RGB images can be represented by an array with three channels, while a batch of images requires an additional batch dimension.

In terms of PyTorch, they commonly represent image batches using the shape:

```math
(B, C, H, W),
```

where $B$ is batch size, $C$ is the number of channels, and $H, W$ are image height and width.

## Calculus
In machine learning, calculus provides the underlying mechanics that neural networks use to adjust their parameters automatically, to better learn their data.

### 2.1 Derivatives
The derivative measures the instantaneous rate of change of a function $f$. For this differentiable function, its derivative is

```math
\lim_{h\to 0}
\frac{f(x+h)-f(x)}{h}.
```
In deep learning, derivatives describe how changes in model parameters affect the loss function.

### 2.2 Partial Derivatives
Partial derivatives extend the derivative from one variable, to many different variables. For a function $`f(x_1, x_2, \ldots, x_n)`$, the partial derivative with respect to $x_i$ measures how the function changes with respect to $x_i$, while the other variables remain fixed.

In terms of notation, it is denoted as

```math
\frac{\partial f}{\partial x_i}
```

### 2.3 Gradients
For a scalar-valued function, the gradient $\nabla$ is a vector consisting of the partial derivatives of said function.

```math
\begin{bmatrix}
\frac{\partial f}{\partial x_1} \
\frac{\partial f}{\partial x_2} \
\vdots \
\frac{\partial f}{\partial x_n}
\end{bmatrix}.
```

The gradient points in the direction of steepest local increase under standard Euclidean geometry. Therefore when the gradient is negative, it provides a direction of local decrease when the gradient is nonzero.

### 2.4 The Chain Rule
For $`y=f(u) \quad u=g(x)`$, in which $`y`$ is a composition of two functions, the chain rule allows us to differentiate $`y`$ in this format

```math
\frac{dy}{du}
\frac{du}{dx}.
```

Within neural network architecture, they consist of multiple composed transformations. With this, backpropagation systematically applies the chain rule to compute derivatives of the loss with respect to trainable parameters.

### 2.5 Gradient Descent
Gradient descent is an iterative optimization method that updates parameters in the direction opposite to the gradient vector. A standard update is defined as:

```math
\boldsymbol{\theta}_t -
\eta\nabla_{\boldsymbol{\theta}}
\mathcal{L}(\boldsymbol{\theta}_t),
```

For $`\eta`$ is the learning rate, which controls the size of each update. an excessively large learning rate $`\eta`$ can cause optimization to be unstable, while a very small learning rate may result in little to no progress.

Between gradients and the gradient descent method in neural network theory, backpropagation computes the gradients of the loss function with respect to the model parameters, while optimizers use those same gradients to update the model parameters.

## Probability and Statistics
When evaluating the predictions, performance, and uncertainty of a model, probability and statistics provide the tools to do this aspect of machine learning.

### 3.1 Probability Distributions
A probability distribution describes how differing probabilities are assigned to possible outcomes of a discrete random variable $X$, described as

```math
P(X=x_i) = p_i,
```

For $`\sum_i p_i = 1`$. In classification, a neural network may produce a probability distribution over possible classes.

### 3.2 Expected Value
The expected value of a discrete random variable is

```math
\sum_i x_u P(X=x_i).
```

In statistical learning, expected values appear when describing the average losses or other quantities over a data distribution.

### 3.3 Variance
Variance measures the expected squared deviation from the mean

```math
\mathbb{E}\left[
(X-\mathbb{E}[X])^2
\right].
```

Variance is useful when analyzing variability in data and comparing results across repeated experiments.

### 3.4 Descriptive Statistics
Common descriptive statistics include the mean, median, standard deviation, and minimum and maximum values. For observations $`x_1, \ldots, x_N`$, the sample mean is:

```math
\frac{1}{N}\sum_{i=1}^{N}x_i.
```

These statistics can help summarize datasets, monitor training behavior, and interpret experimental results.

### 3.5 Classification and Model Evaluation
Classification models assign prediction to discrete categories, Common evaluation metrics include
- **Accuracy**: Proportion of correctly classified observations.
- **Precision**: Proportion of predicted positive observations that are actually positive.
- **Recall**: Proportion of actual positive observations correctly identified.
- **F1 Score**: Harmonic mean of precision and reacll.

For binary classification

```math
\frac{TP}{TP+FP},
```

```math
\frac{TP}{TP+FN},
```

where $TP$, $FP$, and $FN$ denote true positives, false positives, and false negatives, respectively. These ratios are defined when their denominators are nonzero.

These metrics help evaluate model behavior beyond accuracy alone, particularly when class distributions are imbalanced.

## Putting Everything Together
The matematical foundations discussed above work together during neural network training.

A typical supervised learning workflow involves:

1. **Representing data**: Inputs are organized into vectors, matrices, or tensors.
2. **Forward propagation**: The network applies mathematical transformations to produce predictions.
3. **Computing the loss**: A loss function measures disagreement between predictions and targets.
4. **Backpropagation**: Derivatives are calculated using the chain rule.
5. **Optimization**: Model parameters are updated using the computed gradients.
6. **Evaluation**: Statistical metrics are used to assess performance on appropriate held-out data.

These operations form the foundation of training neural networks, including CNNs.

## Final Remarks

Participants are not expected to memorize every mathematical expression in this document. Instead, they should develop enough familiarity with these concepts to recognize their roles in neural network architectures, training algorithms, and experimental evaluation.

Additional mathematical explanations will be introduced throughout the workshop as needed.
