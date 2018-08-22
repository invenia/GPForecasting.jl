# Kernels

Each kernel has its own type and subtypes `Kernel`. Basic algebraic operations are defined
for kernels, such that one can directly operate over them. Some examples:

```julia
julia> k1 = EQ()
EQ()

julia> k2 = RQ(3.0)
RQ(3.0)

julia> k1 + k2
(EQ() + RQ(3.0))

julia> typeof(ans)
GPForecasting.SumKernel

julia> k1 + 3 * k2 + 4
((EQ() + (3 * RQ(3.0))) + (4 * 𝟏))

julia> typeof(ans)
GPForecasting.SumKernel

julia> k1 * k2
(EQ() * RQ(3.0))

julia> typeof(ans)
GPForecasting.ProductKernel

julia> 15 * k1
(15 * EQ())

julia> typeof(ans)
GPForecasting.ScaledKernel
```

In order to better define kernel algebra, we have a `ZeroKernel` type:

```julia
julia> Kernel(0)
𝟎

julia> typeof(ans)
GPForecasting.ZeroKernel

julia> 0 + EQ()
EQ()

julia> 0 * EQ()
𝟎

julia> zero(Kernel)
𝟎
```

Matrix operations are also defined for kernels:

```julia
julia> M = [1 2; 3 4]
2×2 Array{Int64,2}:
 1  2
 3  4

julia> M * EQ()
GPForecasting.Kernel[(1 * EQ()) (2 * EQ()); (3 * EQ()) (4 * EQ())]

julia> typeof(ans)
GPForecasting.MultiKernel

julia> K = MultiKernel([EQ() 0; RQ(5.0) MA(0.5)])
GPForecasting.Kernel[EQ() 𝟎; RQ(5.0) MA(0.5)]

julia> M * K
GPForecasting.Kernel[((1 * EQ()) + (2 * RQ(5.0))) (2 * MA(0.5)); ((3 * EQ()) + (4 * RQ(5.0))) (4 * MA(0.5))]

julia> K * M
GPForecasting.Kernel[(1 * EQ()) (2 * EQ()); ((1 * RQ(5.0)) + (3 * MA(0.5))) ((2 * RQ(5.0)) + (4 * MA(0.5)))]

julia> M * K * M'
GPForecasting.Kernel[((1 * ((1 * EQ()) + (2 * RQ(5.0)))) + (4 * MA(0.5))) ((3 * ((1 * EQ()) + (2 * RQ(5.0)))) + (8 * MA(0.5))); ((1 * ((3 * EQ()) + (4 * RQ(5.0)))) + (8 * MA(0.5))) ((3 * ((3 * EQ()) + (4 * RQ(5.0)))) + (16 * MA(0.5)))]

julia> typeof(ans)
GPForecasting.MultiKernel
```

## Special operations

In order to provide general kernel implementations, we have set up some special operations.

`stretch`, `▷`, `\triangleright`: Stretch the lengthscale of a kernel.

`periodicise`: Make a kernel periodic.

`←`, `\leftarrow`: Define which input is received by the kernel. See below.

## SpecifiedQuantityKernel

Since we might have multiple different features, it is important to define over which ones a given kernel will be computed. For that, we have the `SpecifiedQuantityKernel` type, which acts over a `DataFrame` object. Example:

```julia
julia> df = DataFrame([[1.,2.,3.], [1.,1.,1.]], [:input1, :input2])
3×2 DataFrames.DataFrame
│ Row │ input1 │ input2 │
├─────┼────────┼────────┤
│ 1   │ 1.0    │ 1.0    │
│ 2   │ 2.0    │ 1.0    │
│ 3   │ 3.0    │ 1.0    │

julia> k = EQ()
EQ()

julia> k([1, 2, 3])
3×3 Array{Float64,2}:
 1.0       0.606531  0.135335
 0.606531  1.0       0.606531
 0.135335  0.606531  1.0

julia> k([1, 1, 1])
3×3 Array{Float64,2}:
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> (k ← :input1)(df)
3×3 Array{Float64,2}:
 1.0       0.606531  0.135335
 0.606531  1.0       0.606531
 0.135335  0.606531  1.0

julia> (k ← :input2)(df)
3×3 Array{Float64,2}:
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0

julia> SpecifiedQuantityKernel(Fixed(:input2), k)(df)
3×3 Array{Float64,2}:
 1.0  1.0  1.0
 1.0  1.0  1.0
 1.0  1.0  1.0
```

## Multi-dimensional kernels

Multi-dimensional kernels are implemented in order to allow multi-output GPs. In our convention, assuming a multi-dimensional kernel is represented by a matrix of kernels, `K`, the covariance matrix will be computed in input blocks, i.e., the output will be a block matrix in which each block corresponds to `K` computed at two inputs. In other words, whenever we built a covariance matrix, we loop first over kernels and then over inputs. Example:

```julia
julia> mk = MultiKernel([EQ() 123; 0 55*EQ()])
GPForecasting.Kernel[EQ() (123 * 𝟏); 𝟎 (55 * EQ())]

julia> mk(collect(1:3))
6×6 Array{Float64,2}:
 1.0       123.0      0.606531  123.0     0.135335  123.0
 0.0        55.0      0.0        33.3592  0.0         7.44344
 0.606531  123.0      1.0       123.0     0.606531  123.0
 0.0        33.3592   0.0        55.0     0.0        33.3592
 0.135335  123.0      0.606531  123.0     1.0       123.0
 0.0         7.44344  0.0        33.3592  0.0        55.0
```

## Parameter setting, getting and constraining

Depending on the kernel we are using, certain parameters should be constrained to be positive, bounded etc. This ensures the feasibility of the solutions. If one wants to use the natural constraints for each parameter, it suffices to use the basic constructors, which will automatically apply the necessary constraints:

```julia
julia> k = 5 * EQ()
(5 * EQ())

julia> typeof(k.scale)
GPForecasting.Positive{Int64}

julia> k = 5 + EQ()
(EQ() + (5 * 𝟏))

julia> typeof(k.k2.scale)
GPForecasting.Positive{Int64}

julia> k = MA(0.5)
MA(0.5)

julia> typeof(k.ν)
GPForecasting.Fixed{Float64}
```

In case one desires a behaviour different from the usual one, parameter types can be directly assigned:

```julia
julia> k = Fixed(5) * EQ()
(5 * EQ())

julia> typeof(k.scale)
GPForecasting.Fixed{Int64}

julia> k = Bounded(5, 2, 7) * EQ()
(5 * EQ())

julia> typeof(k.scale)
GPForecasting.Bounded{Int64}
```

For a list of all `Parameter` subtypes, see [Functions](Functions.md).

In order to get the tunable parameters of a kernel, one only needs a simple call:

```julia
julia> k = periodicise(3*(EQ()▷12), 24)
((3 * (EQ() ▷ 12)) ∿ 24)

julia> k[:]
3-element Array{Float64,1}:
 3.17805
 1.09861
 2.48491

julia> k = periodicise(3*(EQ()▷12), Fixed(24))
((3 * (EQ() ▷ 12)) ∿ 24)

julia> k[:]
2-element Array{Float64,1}:
 1.09861
 2.48491
```

Notice that the obtained values don't seem to correspond to the correct values. That happens because parameters all given in the transformed space, where they can be freely optimised, i.e.:

```julia
julia> k = periodicise(3*(EQ()▷12), Fixed(24))
((3 * (EQ() ▷ 12)) ∿ 24)

julia> k[:]
2-element Array{Float64,1}:
 1.09861
 2.48491

julia> exp.(k[:])
2-element Array{Float64,1}:
  3.0
 12.0
```

 Parameter setting can be done via a call to `set`:

 ```julia
 julia> k = periodicise(3*(EQ()▷12), Fixed(24))
 ((3 * (EQ() ▷ 12)) ∿ 24)

 julia> set(k, log.([123, 321]))
 ((123 * (EQ() ▷ 321)) ∿ 24)
```
