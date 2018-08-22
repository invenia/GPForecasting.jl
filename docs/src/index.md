# GPForecasting

[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://doc.invenia.ca/research/GPForecasting.jl/master)
[![Build Status](https://gitlab.invenia.ca/research/GPForecasting.jl/badges/develop/build.svg)](https://gitlab.invenia.ca/research/GPForecasting.jl/commits/master)
[![Coverage](https://gitlab.invenia.ca/research/GPForecasting.jl/badges/develop/coverage.svg)](https://gitlab.invenia.ca/research/GPForecasting.jl/commits/master)

## Data format

### Input
Throughout the package we adopt the convention that a matrix containing `n` input points,
each defined in a `d`-dimensional space, will have size `n`x`d`.

### Output mean
Output means are `Matrix{Float64}` and have a size of `n`x`p`, where `p` is the number of different outputs for a single input point.

### Output covariance and variance
Covariances computed by calling a kernel or by using the `hourly_cov` function will be given
as `np` x `np` matrices, with which block representing all outputs at a given timestamp (see documentation on kernels). In case one uses the `var` function, variances for each output are given in the same shape as the means.

## List of current functionalities

- Compose kernels and means via linear operations.
- Automatically impose constraints over kernel/mean parameters.
- Easily fetch parameter values (in the transformed space).
- Learn the MLE kernel parameters via LBFGS (not implemented for means yet).
- Condition model on observed data.
- Add noise only for observed measurements (see documentation on the Extended Input Space).
- Run on AWS (see documentation on Experiments).
- Execute the optimised version of the basic Linear Mixing Model (with zero mean and same noise for all latent processes).
- Obtain covariance matrices as hourly blocks.


## Simple usage example

```julia
julia> x = collect(0.0:0.01:10.0);

julia> y = sin.(3*x) + 1e-2 * randn(length(x)); # generate synthetic data

julia> k = 0.7 * periodicise(EQ() ▷ 0.1, 2.0) # define kernel
(0.7 * ((EQ() ▷ 0.1) ∿ 2.0))

julia> gp = GP(k) # create GP
GPForecasting.GP{GPForecasting.ScaledKernel,GPForecasting.ScaledMean}((0.0 * 𝟏), (0.7 * ((EQ() ▷ 0.1) ∿ 2.0)))

julia> gp = learn(gp, x, y, objective, trace=false) # optimise parameters
GPForecasting.GP{GPForecasting.ScaledKernel,GPForecasting.ScaledMean}((0.0 * 𝟏), (0.27164985770150735 * ((EQ() ▷ 0.007373677766344863) ∿ 2.0952737850483216)))

julia> pos = condition(gp, x, y) # condition on observations
GPForecasting.GP{GPForecasting.PosteriorKernel,GPForecasting.PosteriorMean}(Posterior((0.27164985770150735 * ((EQ() ▷ 0.007373677766344863) ∿ 2.0952737850483216)), (0.0 * 𝟏)), Posterior((0.27164985770150735 * ((EQ() ▷ 0.007373677766344863) ∿ 2.0952737850483216))))

julia> dist = pos(collect(8.0:0.01:12.0)) # create finite dimensional distribution
GPForecasting.Normal{Array{Float64,1},Array{Float64,2}}([-0.899979, -0.889383, -0.885242, -0.87753, -0.85595, -0.836465, -0.830907, -0.801059, -0.765807, -0.77168  …  -0.961903, -0.899274, -0.911762, -0.918314, -1.00757, -1.02078, -0.983716, -0.981298, -0.998576, -0.957711], [9.99841e-7 -2.86436e-11 … -8.82267e-24 4.81593e-23; -2.86436e-11 9.99841e-7 … -1.08977e-24 5.94859e-24; … ; -8.82267e-24 -1.08977e-24 … 6.86451e-5 -2.45324e-5; 4.81593e-23 5.94859e-24 … -2.45324e-5 6.86451e-5])

julia> sample(dist, 3) # draw samples
401×3 Array{Float64,2}:
 -0.902216  -0.899255  -0.901302
 -0.890751  -0.889292  -0.887488
 -0.883326  -0.884672  -0.883354
 -0.878955  -0.879055  -0.879033
 -0.854238  -0.856038  -0.856209
 -0.83605   -0.836316  -0.835638
 -0.832325  -0.831351  -0.832856
 -0.80226   -0.80043   -0.800944
 -0.764685  -0.766942  -0.765558
 -0.768691  -0.773044  -0.771563
 -0.729669  -0.729625  -0.731016
 -0.692824  -0.693034  -0.691624
 -0.683665  -0.682142  -0.681312
 -0.685151  -0.683834  -0.686734
 -0.679035  -0.678691  -0.678571
 -0.618086  -0.616426  -0.618104
 -0.608868  -0.607701  -0.611255
 -0.584419  -0.585864  -0.586852
 -0.557777  -0.56023   -0.561277
 -0.542663  -0.540791  -0.542121
 -0.500573  -0.501128  -0.501827
 -0.491143  -0.492095  -0.492612
  ⋮
 -0.746323  -0.754712  -0.746486
 -0.780807  -0.778185  -0.765276
 -0.81276   -0.805984  -0.81166
 -0.78738   -0.787478  -0.792376
 -0.806487  -0.790628  -0.808069
 -0.826355  -0.830629  -0.818242
 -0.832517  -0.854101  -0.83656
 -0.826852  -0.81626   -0.819191
 -0.836537  -0.843497  -0.84175
 -0.874248  -0.871165  -0.890342
 -0.928187  -0.928572  -0.911643
 -0.959679  -0.967524  -0.968243
 -0.897242  -0.88479   -0.898077
 -0.924178  -0.90384   -0.908826
 -0.914535  -0.930181  -0.913727
 -1.00056   -1.00386   -1.00097
 -1.03136   -1.01634   -1.0273
 -0.995179  -0.995656  -0.982268
 -0.967596  -0.96638   -0.985257
 -0.997677  -0.994244  -0.993315
 -0.954191  -0.95326   -0.95132
```
