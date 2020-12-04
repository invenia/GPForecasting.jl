# Noisy Observations and Extended Input Space

One might want to add noise to observed measurements, while keeping predictions noiseless. For such task, we have defined `NoiseKernel` and two different `Input` types, `Observed` and `Latent`. Whenever `NoiseKernel` receives two inputs of type `Observed`, it will add noise, while it will not for any other combination of `Inputs`. Example:

```julia
julia> x = [1, 2, 3, 11, 22]
5-element Array{Int64,1}:
  1
  2
  3
 11
 22

julia> y = [1, 2, 3, 4, 5]
5-element Array{Int64,1}:
 1
 2
 3
 4
 5

julia> ox = Observed(x)
GPForecasting.Observed([1, 2, 3, 11, 22])

julia> oy = Observed(y)
GPForecasting.Observed([1, 2, 3, 4, 5])

julia> py = Latent(y)
GPForecasting.Latent([1, 2, 3, 4, 5])

julia> px = Latent(x)
GPForecasting.Latent([1, 2, 3, 11, 22])

julia> nk = GPForecasting.NoiseKernel(EQ(), 12*DiagonalKernel())
GPForecasting.NoiseKernel(GPForecasting.EQ(), 12 * GPForecasting.DiagonalKernel())

julia> nk(px, oy) ≈ nk(ox, py) ≈ nk(px, py)
true

julia> nk(ox, oy) ≈ nk(px, py) + [12 0 0 0 0; 0 12 0 0 0; 0 0 12 0 0; 0 0 0 0 0; 0 0 0 0 0]
true

julia> nk(ox) ≈ nk(px) + [12 0 0 0 0; 0 12 0 0 0; 0 0 12 0 0; 0 0 0 12 0; 0 0 0 0 12]
true
```

NOTE: In case one does not want to work on the extended input space, i.e., wants to have noise either always or never added to the covariances, then one should simply incorporate this to their kernel and work with non-`Input`-typed inputs, e.g.:

```julia
julia> kernel_without_noise = EQ()
GPForecasting.EQ()

julia> kernel_with_noise = EQ() + 12*DiagonalKernel()
GPForecasting.EQ() + 12 * GPForecasting.DiagonalKernel()

julia> x = collect(1.0:5.0)
5-element Array{Float64,1}:
 1.0
 2.0
 3.0
 4.0
 5.0

julia> kernel_without_noise(x)
5×5 Array{Float64,2}:
 1.0          0.606531  0.135335  0.011109  0.000335463
 0.606531     1.0       0.606531  0.135335  0.011109
 0.135335     0.606531  1.0       0.606531  0.135335
 0.011109     0.135335  0.606531  1.0       0.606531
 0.000335463  0.011109  0.135335  0.606531  1.0

julia> kernel_with_noise(x)
5×5 Array{Float64,2}:
 13.0           0.606531   0.135335   0.011109   0.000335463
  0.606531     13.0        0.606531   0.135335   0.011109
  0.135335      0.606531  13.0        0.606531   0.135335
  0.011109      0.135335   0.606531  13.0        0.606531
  0.000335463   0.011109   0.135335   0.606531  13.0
```

Note that `NoiseKernel`s will work properly with `MultiKernel`s. One can even build a `NoiseKernel` by combining a `MultiKernel` and a `DiagonalKernel`, in which case the proper extension of the latter will be performed.

One can even feed `Vector`s of `Input`s and those will be treated in the appropriate space. Example:

```julia
julia> mx = [Latent([1,2,3]), Observed([11, 22]), Latent([15])]
3-element Array{GPForecasting.Input,1}:
 GPForecasting.Latent([1, 2, 3])
 GPForecasting.Observed([11, 22])
 GPForecasting.Latent([15])

julia> nk = GPForecasting.NoiseKernel(EQ(), 12*DiagonalKernel())
GPForecasting.NoiseKernel(GPForecasting.EQ(), 12 * GPForecasting.DiagonalKernel())

julia> nk(mx)
6×6 Array{Any,2}:
 1.0          0.606531     0.135335      1.92875e-22   1.73008e-96  2.74879e-43
 0.606531     1.0          0.606531      2.57676e-18   1.3839e-87   2.00501e-37
 0.135335     0.606531     1.0           1.26642e-14   4.07236e-79  5.38019e-32
 1.92875e-22  2.57676e-18  1.26642e-14  13.0           5.31109e-27  0.000335463
 1.73008e-96  1.3839e-87   4.07236e-79   5.31109e-27  13.0          2.28973e-11
 2.74879e-43  2.00501e-37  5.38019e-32   0.000335463   2.28973e-11  1.0

julia> diag(nk(mx)) ≈ [1.0, 1.0, 1.0, 13.0, 13.0, 1.0]
true
```

In case one feeds inputs to `NoiseKernel` that are not sub-typed from `Input`, it is assumed that one works from an extended input space, i.e., a space in which every point has its corresponding noisy and non-noisy realisation. In this case, our convention is to return noisy and non-noisy values grouped by input, i.e., `[Observed(x[1]), Latent(x[1]), Observed(x[2])...]`. Example:

```julia
julia> x = [1., 2., 3.]
3-element Array{Float64,1}:
 1.0
 2.0
 3.0

julia> nk = GPForecasting.NoiseKernel(EQ(), 12*DiagonalKernel())
GPForecasting.NoiseKernel(GPForecasting.EQ(), 12 * GPForecasting.DiagonalKernel())

julia> nk(x)
6×6 Array{Any,2}:
 13.0       1.0        0.606531  0.606531   0.135335  0.135335
  1.0       1.0        0.606531  0.606531   0.135335  0.135335
  0.606531  0.606531  13.0       1.0        0.606531  0.606531
  0.606531  0.606531   1.0       1.0        0.606531  0.606531
  0.135335  0.135335   0.606531  0.606531  13.0       1.0
  0.135335  0.135335   0.606531  0.606531   1.0       1.0
```

NOTE: Currently, conditioning `NoiseKernel`s over non-`Input`-typed inputs is not implemented as neither is computing `NoiseKernel`s with a mix of `Input`-typed and non-`Input`-typed inputs.
