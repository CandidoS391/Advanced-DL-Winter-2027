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

For two vectors $`x, y \in \mathbf{R}^n`$, the **dot product** is defined as

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

