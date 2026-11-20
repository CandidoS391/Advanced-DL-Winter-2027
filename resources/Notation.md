# Mathematical Notation

This document provides a reference for the mathematical notation used throughout the materials and other custom made references in this project.

Please note that this list is not universally adopted, or in other words the notation you may see here may not be the same across different textbooks, papers, and other outside references.

## 1. General Mathematical Conventions

Unless otherwise specified, we use the following conventions:

| Notation | Meaning |
|---|---|
| $a$ | Scalars |
| $\mathbf{x}, \mathbf{y}$ | Vectors |
| $W, A$ | Matrices |
| $\mathcal{X}$ | Higher-order tensors, when appropriate |
| $\mathbb{R}$ | Set of real numbers |
| $\mathbb{R}^n$ | Space of real-valued vectors with $n$ components |
| $\mathbb{R}^{m \times n}$ | Space of real-valued matrices with $m$ rows and $n$ columns |
| $x_i$ | The $i$-th component of a vector |
| $W_{ij}$ | Entry in row $i$, column $j$ of matrix $W$ |
| $W^\top$ | Transpose of matrix $W$ |
| $\|\mathbf{x}\|_2$ | Euclidean norm of vector $\mathbf{x}$ |

Bold lowercase letters generally represent vectors, uppercase letters represent matrices, and calligraphic letters may represent sets or higher-order tensors.

These conventions may be adjusted when a particular context requires different notation.

## 2. Datasets and Batches

Suppose we have a supervised learning dataset containing $N$ observations.

We write:

$$
\mathcal{D} = \{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\}_{i=1}^{N}.
$$

Here:

| Symbol | Meaning |
|---|---|
| $\mathcal{D}$ | Dataset |
| $N$ | Number of observations |
| $\mathbf{x}^{(i)}$ | Input corresponding to observation $i$ |
| $\mathbf{y}^{(i)}$ | Target corresponding to observation $i$, when represented as a vector |
| $B$ | Mini-batch size |
| $d$ | Number of input features, for vector-valued inputs |
| $C$ | Number of classes in a classification problem |

For a batch of vector-valued inputs, we typically use:

$$
X \in \mathbb{R}^{B \times d}.
$$

Each row corresponds to one observation, and each column corresponds to one feature.

For classification tasks, targets may be represented as integer class labels or one-hot encoded vectors, depending on the loss function and implementation.

## 3. Neural Network Notation

Consider a feedforward neural network with layers indexed by $\ell$.

| Symbol | Meaning |
|---|---|
| $\ell$ | Layer index |
| $L$ | Index of the final layer |
| $\mathbf{a}^{(0)}$ | Input to the neural network |
| $W^{(\ell)}$ | Weight matrix of layer $\ell$ |
| $\mathbf{b}^{(\ell)}$ | Bias vector of layer $\ell$ |
| $\mathbf{z}^{(\ell)}$ | Pre-activation vector of layer $\ell$ |
| $\mathbf{a}^{(\ell)}$ | Activation vector of layer $\ell$ |
| $\sigma^{(\ell)}$ | Activation function of layer $\ell$ |
| $\boldsymbol{\theta}$ | Collection of trainable model parameters |
| $\hat{\mathbf{y}}$ | Model prediction, when represented as a vector |

For a fully connected layer, we use the column-vector convention

```math
\mathbf{z}^{(\ell)}
=
W^{(\ell)}\mathbf{a}^{(\ell-1)}
+
\mathbf{b}^{(\ell)}.
```

The activation is then computed as:

```math
\mathbf{a}^{(\ell)}
=
\sigma^{(\ell)}\left(\mathbf{z}^{(\ell)}\right).
```

If layer $\ell-1$ contains $n_{\ell-1}$ units and layer $\ell$ contains $n_\ell$ units, then:

$$
W^{(\ell)}
\in
\mathbb{R}^{n_\ell \times n_{\ell-1}},
\qquad
\mathbf{b}^{(\ell)}
\in
\mathbb{R}^{n_\ell}.
$$

**Note:** The column-vector convention above describes an individual observation. In PyTorch, batches of observations are commonly stored with the batch dimension first, so the corresponding matrix operation may appear in transposed form.

Superscripts in parentheses, such as $(\ell)$ or $(i)$, indicate layer or observation indices rather than exponentiation.

## 4. Loss Functions and Optimization

The following symbols are used when discussing neural network training.

| Symbol | Meaning |
|---|---|
| $\mathcal{L}$ | Loss function |
| $\boldsymbol{\theta}$ | Trainable model parameters |
| $\eta$ | Learning rate |
| $t$ | Optimization step |
| $\nabla_{\boldsymbol{\theta}}\mathcal{L}$ | Gradient of the loss with respect to model parameters |
| $\partial f/\partial x$ | Partial derivative of $f$ with respect to $x$ |
| $\nabla f$ | Gradient of a scalar-valued function |

A standard gradient descent update is written as:

```math
\boldsymbol{\theta}_{t+1}
=
\boldsymbol{\theta}_t
-
\eta
\nabla_{\boldsymbol{\theta}}
\mathcal{L}(\boldsymbol{\theta}_t).
```

Here, $\mathcal{L}$ represents the objective being optimized, which may be computed over the full dataset or a mini-batch.

For mini-batch training, the gradient is evaluated using the selected mini-batch.

### Common Loss Functions

**Mean Squared Error (MSE):**

For $N$ observations with scalar targets and predictions:

```math
\mathrm{MSE}
=
\frac{1}{N}
\sum_{i=1}^{N}
(y^{(i)}-\hat{y}^{(i)})^2.
```

**Categorical Cross-Entropy:**

For a single observation with $C$ classes:

```math
\mathcal{L}_{\mathrm{CE}}
=
-\sum_{k=1}^{C}
y_k \log(\hat{p}_k),
```

where $y_k$ is the target class indicator and $\hat{p}_k$ is the predicted probability of class $k$.

For a batch, the loss may be averaged across observations.

## 5. Convolutional Neural Networks

CNNs operate on multidimensional arrays, commonly representing images or intermediate feature maps.

### Image and Tensor Dimensions

We use the following symbols:

| Symbol | Meaning |
|---|---|
| $B$ | Batch size |
| $C_{\mathrm{in}}$ | Number of input channels |
| $C_{\mathrm{out}}$ | Number of output channels |
| $H$ | Input height |
| $W$ | Input width |
| $K_H, K_W$ | Kernel height and width |
| $S_H, S_W$ | Vertical and horizontal stride |
| $P_H, P_W$ | Vertical and horizontal padding |
| $H_{\mathrm{out}}, W_{\mathrm{out}}$ | Output height and width |

For PyTorch image tensors, we use the **NCHW convention**, meaning:

$$
X
\in
\mathbb{R}^{B \times C_{\mathrm{in}} \times H \times W}.
$$

The dimensions correspond to:

1. Batch size
2. Channels
3. Height
4. Width

For example, a batch of 32 RGB images with dimensions $64 \times 64$ has shape:

$$
(32, 3, 64, 64).
$$

### Convolutional Kernels

For a standard PyTorch `Conv2d` layer with one group, the kernel tensor has shape:

$$
K
\in
\mathbb{R}^{C_{\mathrm{out}} \times C_{\mathrm{in}} \times K_H \times K_W}.
$$

Each output channel combines information from the input channels using learnable kernel weights.

### Convolution Versus Cross-Correlation

Although deep learning frameworks commonly refer to this operation as *convolution*, standard CNN implementations typically perform **cross-correlation**, meaning the kernel is not spatially reversed.

For a single input channel and a single output channel, with unit stride and no padding, the operation can be expressed as:

```math
Y_{i,j}
=
\sum_{u=0}^{K_H-1}
\sum_{v=0}^{K_W-1}
K_{u,v}X_{i+u,j+v}.
```

This distinction is important when comparing CNN operations with the mathematical definition of discrete convolution.

### Output Dimensions

For a standard 2D convolution or pooling operation, the output height is:

```math
H_{\mathrm{out}}
=
\left\lfloor
\frac{H+2P_H-D_H(K_H-1)-1}{S_H}
+1
\right\rfloor,
```

where $D_H$ is the vertical dilation factor.

Similarly, the output width is:

```math
W_{\mathrm{out}}
=
\left\lfloor
\frac{W+2P_W-D_W(K_W-1)-1}{S_W}
+1
\right\rfloor.
```

These expressions assume symmetric padding, standard floor-based output sizing, and no special output-size adjustment such as ceiling-mode pooling.

## 6. Additional Conventions

To avoid ambiguity, we adopt the following practices:

- **Indexing:** Mathematical expressions may use either zero-based or one-based indexing, depending on context. The indexing convention should be stated when necessary.
- **Dimensions:** Tensor dimensions should be explicitly identified when introducing a new architecture or operation.
- **Superscripts:** Parenthesized superscripts denote indices unless otherwise stated.
- **Functions:** Activation functions and loss functions should be identified when first introduced.
- **Framework differences:** Mathematical notation may differ from the notation used in PyTorch, NumPy, or external references.

## Final Remarks

This reference is intended to support mathematical consistency, not to prescribe a universal notation for deep learning.

Participants are encouraged to consult it when reading the workshop materials and to clearly define their own notation when documenting independent experiments.
