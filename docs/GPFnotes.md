# Notes on GPForecasting

## Package overview

The goal of GPForecasting is to provide a general Gaussian process package. Naturally, its
development is guided by Invenia's specific goals, however, we want to maintain it as general
as possible, such that we don't end up with something that is over-specialised and constrained
to one specific application. As such, the package works for both uni- and multi-dimensional
problems.

## Data conventions

#### Input
Throughout the package we adopt the convention that a matrix containing `n` input points,
each defined in a `d`-dimensional space, will have size `n`x`d`.

#### Output mean
Output means are `Matrix{Float64}` and have a size of `n`x`p`, where `p` is the number of different outputs for a single input point.

#### Output covariance and variance
Covariances computed by calling a kernel or by using the `hourly_cov` function will be given
as `np` x `np` matrices, with which block representing all outputs at a given timestamp (see documentation on kernels). In case one uses the `var` function, variances for each output are given in the same shape as the means.


## Typical workflow

The construction of a GP model starts with the choice of the kernels and of the means. Those
represent our priors, i.e., our initial assumptions about the problem, before accounting for
the data. e.g.:

```julia
julia> using GPForecasting

julia> kernel = EQ()
EQ()

julia> mean = ConstantMean()
𝟏

julia> gp = GP(mean, kernel) # specify both kernel and mean
GPForecasting.GP{GPForecasting.EQ,GPForecasting.ConstantMean}(𝟏, EQ())

julia> gp = GP(kernel) # specify just kernel, zero mean is assumed
GPForecasting.GP{GPForecasting.EQ,GPForecasting.ZeroMean}(𝟎, EQ())
```

In case the kernels or the means have tunable parameters, one might want to use `learn` in
order to tune those. e.g.:

```julia
julia> x_train = collect(0:0.02:4);

julia> y_train = sin.(3.5 .* x_train) .+ 1e-1 .* randn(length(x_train));

julia> gp = GP(periodicise(EQ() ▷ 1.8, 1.0))
GPForecasting.GP{GPForecasting.PeriodicKernel,GPForecasting.ZeroMean}(𝟎, ((EQ() ▷ 1.8) ∿ 1.0))

julia> gp = learn(gp, x_train, y_train, objective, its=10)
Iter     Function value   Gradient norm
     0     4.895386e+07     1.947273e+07
     1     3.295930e+07     8.331219e+07
     2     1.057053e+06     3.152748e+05
     3     1.055342e+06     3.073005e+05
     4     1.011236e+06     6.575358e+04
     5     1.007523e+06     3.360722e+04
     6     1.005326e+06     2.328142e+04
     7     1.003188e+06     2.097326e+04
     8     9.826001e+05     9.569204e+04
     9     9.724007e+05     5.689460e+04
GPForecasting.GP{GPForecasting.PeriodicKernel,GPForecasting.ZeroMean}(𝟎, ((EQ() ▷ 0.77312) ∿ 4.79132))
```

In the case above, notice that `objective` is one of the arguments passed to `learn`. It is
a `Function` object that represents the function which we minimise. In this case, we use
`GPForecasting.objective`, which simply returns the negative logpdf.

Wether we learned parameters or not, it is essential to condition the model on the observed
data, i.e., input the information of the observed data into the model. This is the step
which takes our prior distribution to our posterior distribution (the one that carries
information about the data and that can be used for predictions). e.g.:

```julia
julia> posterior = condition(gp, x_train, y_train)
GPForecasting.GP{GPForecasting.PosteriorKernel,GPForecasting.PosteriorMean}(Posterior(((EQ() ▷ 0.77312) ∿ 4.79132), 𝟎), Posterior(((EQ() ▷ 0.77312) ∿ 4.79132)))
```

Notice that the output of `condition` is simply another `GP` object.

Once we have a posterior distribution, it can be evaluated at any desired points, returning
a `Distributions` object. In our case, it is a `Gaussian`, which represents a matrix-variate
normal distribution. e.g.:

```julia
julia> gaussian = posterior(x_predict)
GPForecasting.Gaussian{Array{Float64,1},Array{Float64,2}}(
μ: [1.23087, 1.29599, 1.35946, 1.4201, 1.47668, 1.52789, 1.57243, 1.60907, 1.63663, 1.65407  …  -0.883977, -0.90217, -0.917027, -0.92861, -0.936927, -0.941923, -0.943485, -0.94145, -0.935613, -0.925746]
Σ: [8.66782e-7 1.1441e-6 … 2.31477e-8 2.26676e-8; 1.1441e-6 1.5451e-6 … 3.60139e-8 3.66234e-8; … ; 2.31477e-8 3.60139e-8 … 8.78069e-8 8.68266e-8; 2.26676e-8 3.66234e-8 … 8.68266e-8 8.80745e-8]
U: Array{Any}(0,0)
)
```

Notice that the `Gaussian` object, being a `Distributions` object, can be operated like one
would normally do with other distributions (it can be `sample`d etc.).

## The `Gaussian` type

The reason we define our own distribution instead of simply using `MvNormal` is that, by
adopting a matrix-variate description, we are able to maintain the distinction between time
and output number (e.g. nodal price) dimensions. This way, it becomes easy to inspect means
and to have block-diagonal covariances. e.g.:

```julia
julia> mean(gaus)
3×4 Array{Int64,2}:
 1  2   4  3
 2  4   8  3
 4  8  16  3
 ```

 Above it is easy to identify that we have 4 outputs and three timestamps (see [data conventions](#data-conventions)). If we had used
 `MvNormal` the output would be a single vector with the values stacked in one of the
 dimensions (depending on the convention adopted), which would be confusing.

 More importantly, our system currently works with hourly covariances. Thus instead of
 dealing with multiple `MvNormal` objects, one can simple use a single `Gaussian` that has a
 `BlockDiagonal` object as covariance matrix. e.g.:

 ```julia
 julia> cov(gaus)
12×12 GPForecasting.OptimisedAlgebra.BlockDiagonal{Float64}:
 3.02469  2.41975   4.83951  0.0      0.0      0.0       0.0      0.0      0.0      0.0       0.0      0.0
 2.41975  6.65432   9.67901  0.0      0.0      0.0       0.0      0.0      0.0      0.0       0.0      0.0
 4.83951  9.67901  21.1728   0.0      0.0      0.0       0.0      0.0      0.0      0.0       0.0      0.0
 0.0      0.0       0.0      1.81481  0.0      0.0       0.0      0.0      0.0      0.0       0.0      0.0
 0.0      0.0       0.0      0.0      3.02469  2.41975   4.83951  0.0      0.0      0.0       0.0      0.0
 0.0      0.0       0.0      0.0      2.41975  6.65432   9.67901  0.0      0.0      0.0       0.0      0.0
 0.0      0.0       0.0      0.0      4.83951  9.67901  21.1728   0.0      0.0      0.0       0.0      0.0
 0.0      0.0       0.0      0.0      0.0      0.0       0.0      1.81481  0.0      0.0       0.0      0.0
 0.0      0.0       0.0      0.0      0.0      0.0       0.0      0.0      3.02469  2.41975   4.83951  0.0
 0.0      0.0       0.0      0.0      0.0      0.0       0.0      0.0      2.41975  6.65432   9.67901  0.0
 0.0      0.0       0.0      0.0      0.0      0.0       0.0      0.0      4.83951  9.67901  21.1728   0.0
 0.0      0.0       0.0      0.0      0.0      0.0       0.0      0.0      0.0      0.0       0.0      1.81481
 ```

 The `BlockDiagonal` type, defined in the module `OptimisedAlgebra`, is powerful in that it
 only stores the blocks in the diagonal of the matrix, not only decreasing allocations, but
 also optimising several important operations, since block diagonal matrices are usually
 easy to work with. We have defined most of the important operations with this kind of
 matrix, including `chol` and some of the specialised Kronecker products (see:
 https://gitlab.invenia.ca/research/oldGPForecasting.jl/merge_requests/81).

 In case one wants to use `MvNormal` type, there are methods such as `MvNormal(d::Gaussian)`
 and `MvNormal(gp::GP, x)`.

## Input types

One might notice that most types are annotated, except for the type of the inputs (usually called `x` around the code). While the outputs (usually called `y`) are
expected to be given either as a `Vector` (in the single-output case) or as a `Matrix` with
timestamps as rows and output number as columns (in accordance to our [Data conventions](#data-conventions)), inputs can take several different types. The main reason
for this is due to our [kernel implementations](#kernels). There are mainly three types of
inputs being used currently:

- `AbstractArray`: this can be either a `Vector`, for the unidimensional case, or an
`AbstractMatrix` for the multidimensional case. In this case, all data is fed to the same
kernel (i.e., in the multidimensional case the kernel has to be able to deal with
multidimensional inputs).

- `DataFrame`: this is used when we want to work with the multi-variate case, for instance,
if we want our model to receive time, load and temperature as inputs. In this case, one
should create a `DataFrame` that has the name of the variables as headers and input it to
a `SpecifiedQuantityKernel` (more on that later), which will pass the correct data to each
of the kernels (as an `AbstractArray`).

- `GPForecasting.Input`: this is a supertype that, currently, encompasses `Latent` and
`Observed`. It only works with kernels of the type `NoiseKernel`. Without getting into GP-related details, the reason for this is that we can
either assume that our data was measured exactly or that it is contaminated with observation
noise. This assumption changes the way we treat the data, thus, in case someone wants to
use both assumptions for different parts of the data, these types are used. For example it might be advisable to assume that the training _outputs_ were measured with noise, but that we want to predict the noiseless outputs. In this case, the training _inputs_ should be marked as `Observed` and the prediction _inputs_ should be marked as `Latent`. The rest will automatically be taken care of by the model. For a more detailed discussion on this, see [here](https://gitlab.invenia.ca/research/GPForecasting.jl/blob/master/docs/src/EIS.md).
It is also worth noting that `Vector{Input}` is a valid input type, in case one wants to mix `Latent` and `Observed` points.

Since these last two types are only broken down into `AbstractArray` at the last point of the calculation (when the inputs are passed to either `NoiseKernel` or to `SpecifiedQuantityKernel` for the last time), the entire pipeline preceding this step has to be clear from type annotations in the input variables (which includes `learn`, `condition`, `credible_interval`, `(gp::GP)(x)`, `logpdf` etc.).

## Parameter handling

An important part of our workflow is using the `learn` function, in order to adjust our priors to the data. Basically, that executes some form of gradient descent optimisation. While means are in general quite flexible, kernels are special functions in that they need
to be positive definite, which imposes certain constraints over their parameters. Moreover,
other parameters might be constrained due to physical reasons or due to the user's choice.
For instance, one might simply not want to optimise a given parameter. With this in mind, we
need an automatic way to annotate and apply constraints to each parameter. There are two main ways of proceeding with this, either by using a numerical optimiser that can have constraints built in, or by performing unconstrained optimisation in a transformed space. We choose the latter, as it is compatible with any optimisation routine we might want to use. NOTE: There are also differences in convergence speed between the two different ways of doing this, but that's beyond the point here.

To make this approach clear, let's look at an example. Assume that our kernel has two parameters, one that needs to be positive, `p1`, and another one that can assume any value, `p2`. If we just plug both values into a numerical optimiser, there is the risk that `p1` may return a negative value (or that the routine may break, as some function becomes ill-defined when computed with `p1` negative). As a simple work-around, we can, instead of optimising in the space defined by `p1` and `p2`, optimise in the space defined by
`log(p1)` and `p2`. As we know, the `log` of a positive number has support over the entire real line, thus, any number that is returned is valid for `log(p1)`, all the while `p2` remains untransformed, as it is not constrained. Naturally, the result of this optimisation has to be transformed back to the original space, which is done via the inverse transform, in this case, `p1 = exp(log(p1))`. This ensures that the resulting value for `p1` will be positive.

In GPForecasting, we have the parameter handling system (defined in src/core/parameter.jl) that takes care of doing this automatically for us. The main concept is the `Parameter` supertype, which wraps all variables that need to be transformed. All transformations occur by calling `pack` (which takes the parameters to the transformed space) and `unpack` (which takes values back to the original space). This way, `pack(x::Real) = x` and `unpack(x, Real) = x`, while `pack(x::Positive) = log(x)` and `unpack(x, Positive) = Positive(exp(x))`(actual implementations are slightly different, this is just a sketch). Since we are going to optimise with respect to these parameters, all of these transformations need be Nabla compatible, as we need their gradients.

## Nodes

Despite the idea behind the `Parameter`s being very simple, its implementation is not. That happens because we might have nested objects that each have their own parameters, so grabbing these parameters and transforming them to the space where we'll perform the optimisation must be recursive. Moreover, it has to be a reversible process, as we will need to get the vector with the optimised parameters back from the optimiser and reconstruct the original object, but with the new values. Another important consideration here is that this part of the package also has to be Nabla-compatible, as it is used during
the numerical optimisations.

The current implementation might very well not be ideal. In case this is getting changed, there are three important assumptions that the rest of the packages makes with regards to it:

- Getting parameters: there is a `get` function that returns a `Vector{Real}` with all the values of the parameters in their transformed space. This has to be recursive, such that, for instance, when calling `get(k)` (also equivalent to `k[:]`) over a nested kernel `k`, the resulting vector must have the transformed parameters (thus applying `pack`) corresponding to every in each kernel of the tree (in some conventional order). Also it is important to note that parameters of the type `Fixed` should not be mapped anywhere, as they are not meant to be optimised (thus the length of the vector can be smaller than the total number of parameters).

- Setting parameters: there is a `set` function that takes the original object, a `Vector{Real}` with values and returns an object of the exact same type as the original one, but with updated parameter values (in the untransformed space, so using `unpack`). This is why we need to have the values in order.

- Being Nabla-compatible: pretty much all, if not actually all, the numerical optimisations in the package will pass through this part of the code, so it must be compatible with Nabla.

Any other implementation that respects these three points should work smoothly with the rest of the package. If our current implementation is slow or memory inefficient, this could be an interesting place to optimise.

## Kernels

A user-friendly explanation of our kernel system can be found [here](https://gitlab.invenia.ca/research/GPForecasting.jl/blob/master/docs/src/Kernels.md).

A few things to keep in mind here are:

- Inputs can have several different types up until they hit the last one of the special wrapping kernels. These special kernels are `SpecifiedQuantityKernel`, which deals with inputs of type `DataFrame`, and `NoiseKernel`, which deals with either untyped inputs ([see here](https://gitlab.invenia.ca/research/GPForecasting.jl/blob/master/docs/src/EIS.md)) or with inputs of type `<:Input` (or vectors thereof).

- Kernels that can wrap other kernels (and are not included in the point above), are agnostic towards the input type they receive.

- Base kernels (that cannot wrap any other kernel), require inputs to be `Union{<:AbstractArray, <:Real}`.

- Most base kernels call `sq_pairwise_dist` during their evaluations, so that function must be Nabla-compatible. Any other type of kernel must also be Nabla compatible (only reason why we were not able to implement the Neural Kernel Networks yet).

- `Kernel`s are `Node`s because we need to get their parameters when optimising the priors.

## Means

Means work pretty much like the kernels. Their entire structure was based on the same principles and their algebra is defined in a similar manner. They also need to be Nabla-compatible and to work as `Node`s.

## Optimisations

We have a separate module called `OptimisedAlgebra` which is essential in making the model attainable to systems of the size we work with. It is pretty much a stand-alone part of the package, and might become it's own thing in the near future. There are two main groups of optimisations there:

- `BlockDiagonal` type: this is a surprising omission in julia. Block diagonal matrices are very common objects in linear algebra and have several nice properties, making their memory allocation much lower and some methods much more efficient. There, we have the basics of the type with several specialised methods that are particularly interesting for us. These are basically `chol` and the other methods that belong to the group below.

- Kronecker product-related optimisations: without getting into many details, the multiple output GP models we utilise lead to several expressions containing Kronecker products. While one can naively compute the products first and then do the other matrix operations, this is highly inefficient (to the point of infeasibility in some of our cases). However, one can exploit the enormous amount of structure present in those expressions in order to directly compute the final result in a much more efficient way, without passing through the expensive intermediate steps. Each optimisation comes from observation of the particular structure of each operation, but all of them have docstrings explaining which expressions they are substituting. Benchmarks easily show their efficiency when used in the intended cases.

## Nabla

Keeping everything Nabla-compatible can be very painstaking, but it is still considerably more efficient than manually implementing all derivatives. The one moment in which Nabla is called is when we call `learn`. That usually leads to a pipeline of the form `learn` -> `get` -> `pack` -> `logpdf` -> `(m::Mean)(x)`/`(k::Kernel)(x, x)` -> `set` -> `unpack`. There might be some hidden methods there, such as if the kernel calls `sq_pairwise_dist`, however, the ones listed above will always be called.

An important is to remember to use `@unionise` whenever making type declarations in types and in methods, since it will automatically compile the corresponding ones that accept `Nabla.Branch` and `Nabla.Leaf`.

## Notes on some internal methods

Some methods are not properly described, since they are not user-facing. Here's some of them:

- `fuse`: this method receives a `Matrix` of matrices (i.e. a block matrix) and returns a regular `Matrix` in which the internal structure has been removed, i.e., it _fuses_ all internal matrices into a single, wrapping one. e.g.:
```julia
julia> A = Matrix(2, 2)
2×2 Array{Any,2}:
 #undef  #undef
 #undef  #undef

julia> A[1, 1] = rand(2, 3)
2×3 Array{Float64,2}:
 0.155405  0.0135963  0.288346
 0.321447  0.806032   0.948718

julia> A[1, 2] = rand(2, 2)
2×2 Array{Float64,2}:
 0.674508  0.254148
 0.199767  0.72349

julia> A[2, 1] = rand(2, 2)
2×2 Array{Float64,2}:
 0.577715  0.493167
 0.684073  0.399099

julia> A[2, 2] = rand(2, 3)
2×3 Array{Float64,2}:
 0.737222  0.190596  0.193544
 0.537312  0.764414  0.671202

julia> A
2×2 Array{Any,2}:
 [0.155405 0.0135963 0.288346; 0.321447 0.806032 0.948718]  [0.674508 0.254148; 0.199767 0.72349]
 [0.577715 0.493167; 0.684073 0.399099]                     [0.737222 0.190596 0.193544; 0.537312 0.764414 0.671202]

julia> GPForecasting.fuse(A)
4×5 Array{Float64,2}:
 0.155405  0.0135963  0.288346  0.674508  0.254148
 0.321447  0.806032   0.948718  0.199767  0.72349
 0.577715  0.493167   0.737222  0.190596  0.193544
 0.684073  0.399099   0.537312  0.764414  0.671202
 ```
 This function is useful because several kernels can be computed in blocks in an efficient way, due to time/space structure.

 - `fuse_equal`: does the same as the above, however, it assumes that the internal blocks have the same size and does the reconstruction more efficiently.

 - `stack`: the behaviour of this function is directly related to our [data conventions](#data-conventions) and to how we compute kernels. As per convention, for multi-output problems, with multiple timestamps, we write the covariance matrices by looping first over outputs, then over timestamps. To illustrate, consider the case of 2 outputs and 2 timestamps. Our covariance matrix will be of the form
 ```julia
 p1t1,p1t1  p1t1,p2t1  p1t1,p1t2  p1t1,p2t2
 p2t1,p1t1  p2t1,p2t1  p2t1,p1t2  p2t1,p2t2
 p1t2,p1t1  p1t2,p2t1  p1t2,p1t2  p1t2,p2t2
 p2t2,p1t1  p2t2,p2t1  p2t2,p1t2  p2t2,p2t2
 ```
 where `pi` means the `ith` output and `tj` means the `jth` timestamp. However, it is frequently convenient to compute covariances per output for all timestamps at once. What `stack` does is take several blocks of the form
 ```julia
 p1t1 ... p1tn
 .    .     .
 .     .    .
 .      .   .
 pmt1 ... pmtn
 ```
 and _stack_ them in the conventional form.

 In a similar way that `stack` takes in matrices corresponding to covariances of individual outputs and intercalates them such that, in the result, we loop over outputs before looping over time, it can also take a vector corresponding to means of different outputs and intercalate them such that we first loop over outputs, i.e., assume the means of two outputs, for three timestamps. A vector with the individual means would be of the form:
 ```julia
 [[p1t1, p1t2, p1t3], [p2t1, p2t2, p2t3]]
 ```
 By calling stack over that, we get
 ```julia
 [p1t1, p2t1, p1t2, p2t2, p1t3, p2t3]
 ```

 In summary, `stack` groups results from different outputs into objects that first loop over time and then over output, as is our convention.

 - `unstack`: although `stack` puts the elements in the conventional order, it does not use our conventional shape for means. That is due to how computations are done: the covariance matrix is a `np` x `np` matrix (see [data conventions](#data-conventions)), thus, it needs an `np`-long vector such sizes compute when doing linear algebra operations. However, `unstack` takes a stacked mean and reshapes it such that it corresponds to `p` outputs. If we call `unstack` on the last example above we get
 ```julia
 [p1t1 p2t1; p1t2 p2t2; p1t3 p2t3]
 ```

 - `hourly_cov`: instead of computing the full covariance matrix, computes only the hourly blocks, i.e., the covariance between each input for when the hours are the same. Returns a `BlockDiagonal` matrix.

 - `_eskmu_fill_diag!`: fills the diagonal of the result of `eye_sum_kron_M_ut`.

 - `_eskmu_fill_triu!`: fills the upper triangle of the result of `eye_sum_kron_M_ut`.

 - `_eskmu_fill_triu_Lt!`: fills the lower triangle of the result of `eye_sum_kron_M_ut`.
