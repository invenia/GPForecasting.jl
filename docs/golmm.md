## GOLMM Kernel
The final form of the “multi-kernel” captures grouping structure between nodes in a separable kernel by incorporating grouping structure in the mixing matrix.

For each of the output time-series $`i \in \{1 \ldots p \}`$, let us introduce a learnable _group embedding_ $`h_i \in \reals^\ell`$ (where $`\ell \geq 1`$). These embeddings can be thought of as coordinates of that time-series in some space where similar series are located close by.

To express this mathematically, we define a _group kernel_ $`k_\theta: \reals^\ell, \reals^\ell \rightarrow \reals`$ ($`\theta`$ being learnable hyperparameters) which we apply to the the group embeddings $`h_{i = 1}^p`$ to construct a Gram covariance matrix for the outputs:
```math
\mathbf{K} =
\begin{bmatrix}
k(h_1, h_1) & \cdots & k(h_1, h_p) \\
\vdots & \ddots & \vdots \\
k(h_p, h_1) & \cdots & k(h_p, h_p) \\
\end{bmatrix}
```

Now, we perform eigendecomposition of this matrix:
```math
K = U S U^T
```
and define the mixing matrix $`H`$ using the $`m`$ dominant eigenvectors
```math
H = U_{\ldots, 1\ldots m} ~ S_{1\ldots m, 1\ldots m}
```
which are orthogonal, and hence compatible with OLMM, by construction.

Intuitively, the above procedure can be understood as kernelised PCA in the learnable $`h`$-space of group embeddings.

__NB__. The current implementation of `GOLMMKernel` in GPForecasting.jl!131 uses $`h \in \reals`$.
