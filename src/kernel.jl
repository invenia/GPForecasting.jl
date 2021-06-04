#########################################################
# Default kernel behaviour:
# k(x) = k(x, x), as a definition for k(x).
# isa(k(x, y), AbstractMatrix) == true, for all x, y.
# k(x, y) == transpose(k(y, x))
#########################################################

"""
elwise(k::Kernel, x, y)

Compute the value of the kernel elementwise for `x` and `y`. Naturally, they must have the
same number of points.

elwise(k::Kernel, x)

Compute the value of the kernel elementwise for `x`, i.e., the variance.
"""
function elwise(k::Kernel, x, y)
    size(x) != size(y) && throw(DimensionMismatch("`x` and `y` must be of same size."))
    return [k(x[i, :], y[i, :])[1] for i in 1:size(x, 1)]
end
# I know the elegant solution below would be to use multiple dispatch, but that leads to
# method ambiguities that can only be solved by implementing a bunch of extra methods.
# So, this is a much more convenient solution.
# This is one example of a place where having method precedence would be very handy.
function elwise(k::Kernel, x)
    if !isa(x, AbstractDataFrame)
        return [k(x[i, :])[1] for i in 1:size(x, 1)]
    else
        return [k(DataFrame(r))[1] for r in eachrow(x)]
    end
end

Statistics.var(k::Kernel, x) = elwise(k, x)
Statistics.var(k::Kernel, x::Vector{Input}) = reduce(vcat, broadcast(c -> var(k, c), x))
# function Statistics.var(k::Kernel, x::Input)
#     return reduce(vcat, broadcast(c -> var(k, typeof(x)(c)), x.val))
# end

Base.size(k::Kernel, i::Int) = i < 1 ? BoundsError() : 1

# Hacky method to detect NoiseKernels
"""
    is_not_noisy(k::Kernel)

Detect if there is a `NoiseKernel` at any part of a composite kernel, in which case it
returns `true`. This is useful for dealing with `Latent` and `Observed` inputs.
"""
is_not_noisy(x) = true
function is_not_noisy(k::Kernel)
    fields = fieldnames(typeof(k))
    isempty(fields) && return true
    return prod([is_not_noisy(getfield(k, f)) for f in fields])
end

"""
    hourly_cov(k::Kernel, x)

Return a sparse matrix with only the variances (at the diagonal).

    hourly_cov(k::MultiOutputKernel, x)

Return a block-diagonal sparse matrix with the covariances between all outputs for each
given input value, i.e., does not compute covariances for different values of `x`.
"""
hourly_cov(k::Kernel, x) = sparse(Diagonal(var(k, x)))

"""
    isMulti(k::Kernel)

True if `k` is a multi output kernel, false otherwise. This is useful because chains of
`SumKernel`s and/or `ScaledKernel`s may hide such information.
"""
isMulti(k::Kernel) = false

"""
    PosteriorKernel <: Kernel

Posterior kernel for a GP.

# Fields:
- `k`: prior kernel.
- `x`: points over which `k` was conditioned.
- `U`: Cholesky decomposition of the covariance matrix.

# Methods:
    (k::PosteriorKernel)(x, y)

Build covariance matrix between `x` and `y`.

    (k::PosteriorKernel)(x) = k(x, x)
"""
mutable struct PosteriorKernel <: Kernel
    k::Kernel
    x
    U
    PosteriorKernel(k, x, U) = new(k, Fixed(x), Fixed(U))
end
Base.show(io::IO, k::PosteriorKernel) = print(io, "Posterior(", k.k, ")")
isMulti(k::PosteriorKernel) = isMulti(k.k)
function (k::PosteriorKernel)(x)
    U = unwrap(k.U)
    xd = unwrap(k.x)
    z = k.k(x, xd) / U
    return k.k(x) .- z * z'
end
function elwise(k::PosteriorKernel, x)
    U = unwrap(k.U)
    xd = unwrap(k.x)
    z = k.k(x, xd) / U
    return elwise(k.k, x) .- sum(z .* z, dims=2)
end
function (k::PosteriorKernel)(x, y)
    U = unwrap(k.U)
    xd = unwrap(k.x)
    return k.k(x, y) .- (k.k(x, xd) / U) * (U' \ k.k(xd, y))
end
function elwise(k::PosteriorKernel, x, y)
    size(x) != size(y) && throw(DimensionMismatch("`x` and `y` must be of same size."))
    U = unwrap(k.U)
    xd = unwrap(k.x)
    return elwise(k.k, x, y) .- sum((k.k(x, xd) / U) .* (U' \ k.k(xd, y))', dims=2)
end

"""
    TitsiasPosteriorKernel <: Kernel

Posterior kernel for a sparse GP under Titsias' approximation. See "Variational
Learning of Inducing Variables in Sparse Gaussian Processes".

This kernel works with `Observed` and `Latent` input types, similarly to `NoiseKernel`s.
Untyped inputs will return values in the extended space, with noisy and denoised values.
"""
mutable struct TitsiasPosteriorKernel <: Kernel
    k::Kernel
    Xm
    Uz
    Umm
    σ²
    function TitsiasPosteriorKernel(k, Xm, Uz, Umm, σ²)
        return new(k, Fixed(Xm), Fixed(Uz), Fixed(Umm), Fixed(σ²))
    end
end

function _titsposkern(k::TitsiasPosteriorKernel, x)
    Xm = unwrap(k.Xm)
    Uz = unwrap(k.Uz)
    Umm = unwrap(k.Umm)

    Kx = k.k(x)
    Kmx = k.k(Xm, x)
    sqrt₁ = Umm' \ Kmx
    sqrt₂ = Uz' \ Kmx
    return Kx .- (sqrt₁' * sqrt₁) .+ (sqrt₂' * sqrt₂)
end
function (k::TitsiasPosteriorKernel)(x::Observed)
    xx = is_not_noisy(k.k) ? x.val : _Observed(x.val)
    return _titsposkern(k, xx) + unwrap(k.σ²) * I
end
function (k::TitsiasPosteriorKernel)(x::Latent)
    xx = is_not_noisy(k.k) ? x.val : x
    return _titsposkern(k, xx)
end
function (k::TitsiasPosteriorKernel)(x)
    notice(LOGGER,
        """
            Working on the extended input space. Output will be two dimensional,
            corresponding to the noisy and denoised predictions. To compute only the
            noisy (denoised) predictions, please wrap your input in `Observed` (`Latent`).
        """
    )
    # We need to first create an empty container and then populate it, otherwise, julia
    # automatically merges all matrices into one, and then `stack` won't work.
    tmp = Matrix{Matrix{Real}}(undef, 2, 2)
    tmp[1, 1] = k(Observed(x))
    tmp[1, 2] = tmp[2, 1] = tmp[2, 2] = k(Latent(x))
    return stack(tmp)
end

function _titselwise(k::TitsiasPosteriorKernel, x)
    Xm = unwrap(k.Xm)
    Uz = unwrap(k.Uz)
    Umm = unwrap(k.Umm)

    kx = elwise(k.k, x)
    Kxm = k.k(x, Xm)
    sqrt₁ = Kxm / Umm
    sqrt₂ = Kxm / Uz
    return kx .- sum(sqrt₁ .* sqrt₁, dims=2) .+ sum(sqrt₂ .* sqrt₂, dims=2)
end
function elwise(k::TitsiasPosteriorKernel, x::Observed)
    xx = is_not_noisy(k.k) ? x.val : _Observed(x.val)
    return _titselwise(k, xx) .+ unwrap(k.σ²)
end
function elwise(k::TitsiasPosteriorKernel, x::Latent)
    xx = is_not_noisy(k.k) ? x.val : x
    return _titselwise(k, xx)
end
function elwise(k::TitsiasPosteriorKernel, x)
    notice(LOGGER,
        """
            Working on the extended input space. Output will be two dimensional,
            corresponding to the noisy and denoised predictions. To compute only the
            noisy (denoised) predictions, please wrap your input in `Observed` (`Latent`).
        """
    )
    tmp = [elwise(k, Observed(x)) elwise(k, Latent(x))]
    return
end

function _titsposkern(k::TitsiasPosteriorKernel, x, y)
    Xm = unwrap(k.Xm)
    Uz = unwrap(k.Uz)
    Umm = unwrap(k.Umm)

    Kxy = k.k(x, y)
    Kxm = k.k(x, Xm)
    Kmy = k.k(Xm, y)

    return Kxy .- (Kxm / Umm) * (Umm' \ Kmy) .+ (Kxm / Uz) * (Uz' \ Kmy)
end
function (k::TitsiasPosteriorKernel)(x::Input, y::Input)
    xx, yy = is_not_noisy(k.k) ? (x.val, y.val) : (x, y)
    return _titsposkern(k::TitsiasPosteriorKernel, xx, yy)
end
function (k::TitsiasPosteriorKernel)(x::Observed, y::Observed)
    noise = unwrap(k.σ²) * DiagonalKernel()(x.val, y.val)
    xx, yy = is_not_noisy(k.k) ? (x.val, y.val) : (_Observed(x.val), _Observed(y.val))
    return _titsposkern(k::TitsiasPosteriorKernel, xx, yy) + noise
end
function (k::TitsiasPosteriorKernel)(x, y)
    notice(LOGGER,
        """
            Working on the extended input space. Output will be two dimensional,
            corresponding to the noisy and denoised predictions. To compute only the
            noisy (denoised) predictions, please wrap your input in `Observed` (`Latent`).
        """
    )
    # We need to first create an empty container and then populate it, otherwise, julia
    # automatically merges all matrices into one, and then `stack` won't work.
    tmp = Matrix{Matrix{Real}}(undef, 2, 2)
    tmp[1, 1] = k(Observed(x), Observed(y))
    tmp[1, 2] = tmp[2, 1] = tmp[2, 2] = k(Latent(x), Latent(y))
    return stack(tmp)
end

function _titselwise(k::TitsiasPosteriorKernel, x, y)
    size(x) != size(y) && throw(DimensionMismatch("`x` and `y` must be of same size."))
    Xm = unwrap(k.Xm)
    Uz = unwrap(k.Uz)
    Umm = unwrap(k.Umm)

    kxy = elwise(k.k, x, y)
    Kxm = k.k(x, Xm)
    Kmy = k.k(Xm, y)
    t₁ = sum((Kxm / Umm) .* (Umm' \ Kmy), dims=2)
    t₂ = sum((Kxm / Uz) .* (Uz' \ Kmy), dims=2)
    return kxy .- t₁ .+ t₂ + unwrap(k.σ²)
end
function elwise(k::TitsiasPosteriorKernel, x::Input, y::Input)
    xx, yy = is_not_noisy(k.k) ? (x.val, y.val) : (x, y)
    return _titselwise(k, xx, yy)
end
function elwise(k::TitsiasPosteriorKernel, x::Observed, y::Observed)
    xx, yy = is_not_noisy(k.k) ? (x.val, y.val) : (_Observed(x.val), _Observed(y.val))
    return _titselwise(k, xx, yy) .+ unwrap(k.σ²)
end
function elwise(k::TitsiasPosteriorKernel, x, y)
    notice(LOGGER,
        """
            Working on the extended input space. Output will be two dimensional,
            corresponding to the noisy and denoised predictions. To compute only the
            noisy (denoised) predictions, please wrap your input in `Observed` (`Latent`).
        """
    )
    tmp = [elwise(k, Observed(x), Observed(y)) elwise(k, Latent(x), Latent(y))]
    return
end


"""
    EQ <: Kernel

Squared exponential kernel. Computes exp((-1/2) * |x - x′|²).
"""
struct EQ <: Kernel end
(::EQ)(x, y) = exp.((-0.5) .* sq_pairwise_dist(x, y))
(k::EQ)(x) = k(x, x)
elwise(k::EQ, x, y) = exp.((-0.5) .* sq_elwise_dist(x, y))
elwise(k::EQ, x) = ones(size(x, 1))

const ArrayOrReal = Union{Wrapped{<:AbstractArray{<:Real}}, Wrapped{<:Real}}

"""
    RQ <: Kernel

Rational quadratic kernel. Computes (1 + ((x - x′)² / (2α)))^α.
"""
mutable struct RQ <: Kernel
    α
    RQ(α) = isconstrained(α) ? new(α) : new(Positive(α))
end
function (k::RQ)(x::ArrayOrReal, y::ArrayOrReal)
    return (1.0 .+ (sq_pairwise_dist(x, y) ./ (2.0 * unwrap(k.α)))) .^ (-unwrap(k.α))
end
(k::RQ)(x::ArrayOrReal) = k(x, x)
function elwise(k::RQ, x::ArrayOrReal, y::ArrayOrReal)
    return (1.0 .+ (sq_elwise_dist(x, y) ./ (2.0 * unwrap(k.α)))) .^ (-unwrap(k.α))
end
elwise(k::RQ, x::ArrayOrReal) = ones(size(x, 1))

"""
    SimilarHourKernel <: Kernel
    SimilarHourKernel(hour_deltas, coeffs) -> SimilarHourKernel
"""
mutable struct SimilarHourKernel <: Kernel
    hdeltas::Fixed{Int}
    coeffs

    function SimilarHourKernel(hdeltas::Union{Fixed{Int}, Int}, coeffs)
        unwrap(hdeltas) > 24 && throw(ArgumentError("Can't have more than 24 hour deltas."))
        unwrap(hdeltas) < 1 && throw(ArgumentError("Need at least one hour delta."))
        length(unwrap(coeffs)) != unwrap(hdeltas) && throw(DimensionMismatch(
            "The number of coefficients must be the same as the number of hour deltas."
        ))
        hd = isconstrained(hdeltas) ? hdeltas : Fixed(hdeltas)
        cs = isconstrained(coeffs) ? coeffs : Positive(coeffs)
        return new(hd, cs)
    end
end
function (k::SimilarHourKernel)(x::ArrayOrReal, y::ArrayOrReal)
    δ(x) = isapprox(x, 0.0, atol=1e-15) ? 1 : 0  # delta function
    d = pairwise_dist(x, y) .% 24
    cs = unwrap(k.coeffs)
    hd = unwrap(k.hdeltas)
    K = cs[1] * δ.(d)
    if length(cs) > 1
        return K .+ sum(cs[i + 1] * (δ.(d .- (24 - i)) + δ.(d .- i)) for i in 1:(hd - 1))
    else
        return K
    end
end
(k::SimilarHourKernel)(x::ArrayOrReal) = k(x, x)
function Base.show(io::IO, k::SimilarHourKernel)
    cs = unwrap(k.coeffs)
    print(io, cs[1], "*δ(0)")
    for i in 1:(unwrap(k.hdeltas) - 1)
        print(io, " + ", cs[i + 1], "*δ($i)")
    end
end


"""
    MA <: Kernel
    MA(ν::Union{Fixed, Real}) -> MA

    Matérn kernel. Implemented only for `ν ∈ (1/2, 3/2, 5/2)`.
"""
struct MA <: Kernel
    ν::Fixed
end
MA(n::Real) = MA(Fixed(n))

function (k::MA)(x::ArrayOrReal, y::ArrayOrReal)
    d = pairwise_dist(x, y)
    if unwrap(k.ν) ≈ 1/2
        return exp.(-d)
    elseif unwrap(k.ν) ≈ 3/2
        return (1 .+ √3 .* d) .* exp.(-√3 .* d)
    elseif unwrap(k.ν) ≈ 5/2
        return (1 .+ √5 .* d + 5/3 .* d.^2) .* exp.(-√5 .* d)
    else
        throw(ArgumentError("$(unwrap(k.ν)) is not a supported value for Matérn kernels."))
    end
end
(k::MA)(x::ArrayOrReal) = k(x, x)

function elwise(k::MA, x::ArrayOrReal, y::ArrayOrReal)
    d = elwise_dist(x, y)
    if unwrap(k.ν) ≈ 1/2
        return exp.(-d)
    elseif unwrap(k.ν) ≈ 3/2
        return (1 .+ √3 .* d) .* exp.(-√3 .* d)
    elseif unwrap(k.ν) ≈ 5/2
        return (1 .+ √5 .* d + 5/3 .* d.^2) .* exp.(-√5 .* d)
    else
        throw(ArgumentError("$(unwrap(k.ν)) is not a supported value for Matérn kernels."))
    end
end
elwise(k::MA, x::ArrayOrReal) = ones(size(x, 1))

"""
    RootLog <: Kernel
    RootLog() -> RootLog

Kernel that computes ``(1/|x - y|) * log(1 + |x - y|)``.
"""
struct RootLog <: Kernel end
function (k::RootLog)(x::ArrayOrReal, y::ArrayOrReal)
    d = pairwise_dist(x, y)
    # This expression here is just to make sure that we get the correct limit when d → 0
    return (log.(max.(d, 1e-8) .+ 1) ./ max.(d, 1e-8)) .+ (1 - 1e8 * log(1 + 1e-8))
end
(k::RootLog)(x::ArrayOrReal) = k(x, x)
function elwise(k::RootLog, x::ArrayOrReal, y::ArrayOrReal)
    d = elwise_dist(x, y)
    # This expression here is just to make sure that we get the correct limit when d → 0
    return (log.(max.(d, 1e-8) .+ 1) ./ max.(d, 1e-8)) .+ (1 - 1e8 * log(1 + 1e-8))
end
elwise(k::RootLog, x::ArrayOrReal) = ones(size(x, 1))

"""
    BinaryKernel <: Kernel
    BinaryKernel(Θ₁, Θ₂, Θ₃=Fixed(0)) -> BinaryKernel

Kernel for binary inputs.

Has three possible outcomes: `Θ₁` if `x = y = 1`, `Θ₂` if `x = y = 0` and `Θ₃` if `x ≠ y`.
Naturally, this only accepts unidimensional inputs.
"""
mutable struct BinaryKernel <: Kernel
    Θ₁
    Θ₂
    Θ₃
    BinaryKernel(Θ₁::Parameter, Θ₂::Parameter, Θ₃::Parameter=Fixed(0)) = new(Θ₁, Θ₂, Θ₃)
end

@unionise function (k::BinaryKernel)(x::AbstractArray{<:Integer}, y::AbstractArray{<:Integer})
    return unwrap(k.Θ₁) .* (x .≈ y' .≈ 1) .+
        unwrap(k.Θ₂) .* (x .≈ y' .≈ 0) .+
        unwrap(k.Θ₃) .* (x .!= y')
end
@unionise (k::BinaryKernel)(x::AbstractArray{<:Integer}) = k(x, x)

function BinaryKernel(a::Real, b::Real, c::Real)
    pa = Positive(a)
    pb = Positive(b)
    return BinaryKernel(
        pa,
        pb,
        Bounded(
            c,
            DynamicBound((x, y) -> -sqrt(x * y), [pa, pb]),
            DynamicBound((x, y) -> sqrt(x * y), [pa, pb]),
        )
    )
end
BinaryKernel(a::Real, b::Real) = BinaryKernel(Positive(a), Positive(b))

"""
    ScaledKernel <: Kernel
    ScaledKernel(scale, k::Kernel)

Result from the multiplication of a kernel `k` by a `Number` or `Parameter`.
Scales the kernel variance.
"""
mutable struct ScaledKernel <: Kernel
    scale
    k::Kernel
end

isMulti(k::ScaledKernel) = isMulti(k.k)

function Base.:*(x, k::Kernel)
    isconstrained(x) && return ScaledKernel(x, k)
    unwrap(x) ≈ zero(unwrap(x)) && return ZeroKernel()
    return ScaledKernel(Positive(x), k)
end
function Base.:*(x, k::ScaledKernel)
    isconstrained(x) && return ScaledKernel(x, k)
    unwrap(x) * unwrap(k.scale) ≈ zero(unwrap(x)) && return ZeroKernel()
    if (isconstrained(k.scale) && !isa(k.scale, Positive))
        return ScaledKernel(Positive(x), k)
    else
        return ScaledKernel(Positive(unwrap(x) * unwrap(k.scale)), k.k)
    end
end
Base.:*(k::Kernel, x) = x * k

function Base.:+(k1::Kernel, k2::ScaledKernel)
    return unwrap(k2.scale) ≈ zero(unwrap(k2.scale)) ? k1 : SumKernel(k1, k2)
end
function Base.:+(k1::ScaledKernel, k2::Kernel)
    return unwrap(k1.scale) ≈ zero(unwrap(k1.scale)) ? k2 : SumKernel(k1, k2)
end
function Base.:+(k1::ScaledKernel, k2::ScaledKernel)
    if unwrap(k1.scale) ≈ zero(unwrap(k1.scale))
        return k2
    elseif unwrap(k2.scale) ≈ zero(unwrap(k2.scale))
        return k1
    else
        return SumKernel(k1, k2)
    end
end

(k::ScaledKernel)(x, y) = unwrap(k.scale) .* k.k(x, y)
(k::ScaledKernel)(x) = k(x, x)

elwise(k::ScaledKernel, x, y) = unwrap(k.scale) .* elwise(k.k, x, y)
elwise(k::ScaledKernel, x) = elwise(k, x, x)

Base.show(io::IO, k::ScaledKernel) = print(io, "(", k.scale, " * ", k.k, ")")

"""
    StretchedKernel(l, k::Kernel) -> StretchedKernel

Represent any kernel `k` with its length scale stretched to be `l`.

See also: [`stretch`](@ref)
"""
mutable struct StretchedKernel <: Kernel
    stretch
    k::Kernel
end

isMulti(k::StretchedKernel) = isMulti(k.k)

"""
    k::Kernel ▷ l -> StretchedKernel
    stretch(k::Kernel, l) -> StretchedKernel

Create a [`StretchedKernel`](@ref) by stretching kernel k's length scale to `l`.
"""
stretch(k::Kernel, x) = StretchedKernel(isconstrained(x) ? x : Positive(x), k)
function stretch(k::StretchedKernel, x)
    return StretchedKernel(
        isconstrained(x) ? x .* unwrap(k.stretch) : Positive(x .* unwrap(k.stretch)),
        k.k
    )
end
const ▷ = stretch

function (k::StretchedKernel)(x, y)
    lscale = unwrap(k.stretch)'

    # This condition should only be met in case the input space got extended by `periodicise`
    # If the user forces this to trigger by feeding a length scale with the wrong length,
    # that is his fault.
    if length(lscale) > 1 && length(lscale) == size(x, 2) / 2
        return k.k(x ./ hcat(lscale, lscale), y ./ hcat(lscale, lscale))
    else
        return k.k(x ./ lscale, y ./ lscale)
    end
end
(k::StretchedKernel)(x) = k(x, x)

function elwise(k::StretchedKernel, x, y)
    lscale = unwrap(k.stretch)'

    # This condition should only be met in case the input space got extended by `periodicise`
    # If the user forces this to trigger by feeding a length scale with the wrong length,
    # that is his fault.
    if length(lscale) > 1 && length(lscale) == size(x, 2) / 2
        return elwise(k.k, x ./ hcat(lscale, lscale), y ./ hcat(lscale, lscale))
    else
        return elwise(k.k, x ./ lscale, y ./ lscale)
    end
end
elwise(k::StretchedKernel, x) = elwise(k, x, x)

# This is a simplification.; in some case stretch(k, l) != StretchedKernel(l, k).
Base.show(io::IO, k::StretchedKernel) = print(io, "(", k.k, " ▷ ", k.stretch, ")")

"""
    SumKernel <: Kernel
    SumKernel(k1::Kernel, k2::Kernel) -> SumKernel

Kernel built by adding two kernels, `k1` and `k2`.
"""
mutable struct SumKernel <: Kernel
    k1::Kernel
    k2::Kernel
end

Base.:+(k1::Kernel, k2::Kernel) = SumKernel(k1, k2)

(k::SumKernel)(x, y) = k.k1(x, y) .+ k.k2(x, y)
(k::SumKernel)(x) = k(x, x)

elwise(k::SumKernel, x, y) = elwise(k.k1, x, y) .+ elwise(k.k2, x, y)
elwise(k::SumKernel, x) = elwise(k, x, x)

# This leads to prettier-printing in most cases, but it is a simplification;
# in some cases `k1 + k2 != SumKernel(k1, k2)` e.g. when `k1 = ZeroKernel()`
Base.show(io::IO, k::SumKernel) = print(io, "(", k.k1, " + ", k.k2, ")")
isMulti(k::SumKernel) = isMulti(k.k1) || isMulti(k.k2)

"""
    ProductKernel <: Kernel
    ProductKernel(k1::Kernel, k2::Kernel) -> ProductKernel

Kernel built by multiplying two kernels, `k1` and `k2`.
"""
mutable struct ProductKernel <: Kernel
    k1::Kernel
    k2::Kernel
end

Base.:*(k1::Kernel, k2::Kernel) = ProductKernel(k1, k2)
function Base.:*(k1::Kernel, k2::ScaledKernel)
    return unwrap(k2.scale) ≈ zero(unwrap(k2.scale)) ? zero(Kernel) : ProductKernel(k1, k2)
end
function Base.:*(k1::ScaledKernel, k2::Kernel)
    return unwrap(k1.scale) ≈ zero(unwrap(k1.scale)) ? zero(Kernel) : ProductKernel(k1, k2)
end
function Base.:*(k1::ScaledKernel, k2::ScaledKernel)
    if unwrap(k1.scale) ≈ zero(unwrap(k1.scale))
        return zero(Kernel)
    elseif unwrap(k2.scale) ≈ zero(unwrap(k2.scale))
        return zero(Kernel)
    else
        return ProductKernel(k1, k2)
    end
end

(k::ProductKernel)(x, y) = k.k1(x, y) .* k.k2(x, y)
(k::ProductKernel)(x) = k(x, x)

elwise(k::ProductKernel, x, y) = elwise(k.k1, x, y) .* elwise(k.k2, x, y)
elwise(k::ProductKernel, x) = elwise(k, x, x)

# This leads to prettier-printing in most cases, but it is a simplification;
# in some cases `k1 * k2 != ProductKernel(k1, k2)` e.g. when `k1 = ZeroKernel()`
Base.show(io::IO, k::ProductKernel) = print(io, "(", k.k1, " * ", k.k2, ")")

isMulti(k::ProductKernel) = isMulti(k.k1) || isMulti(k.k2)

"""
    PeriodicKernel <: Kernel
    PeriodicKernel(period, k::Kernel) -> PeriodicKernel

Kernel built by defining a period `period` for kernel `k`.
"""
mutable struct PeriodicKernel <: Kernel
    T::Parameter  # `T` is common nomenclature for period, and makes the code a little nicer
    k::Kernel
end

function (k::PeriodicKernel)(x, y)
    px = [cos.(2π .* x ./ unwrap(k.T)') sin.(2π .* x ./ unwrap(k.T)')]
    py = [cos.(2π .* y ./ unwrap(k.T)') sin.(2π .* y ./ unwrap(k.T)')]
    return k.k(px, py)
end
(k::PeriodicKernel)(x) = k(x, x)

function elwise(k::PeriodicKernel, x, y)
    px = [cos.(2π .* x ./ unwrap(k.T)') sin.(2π .* x ./ unwrap(k.T)')]
    py = [cos.(2π .* y ./ unwrap(k.T)') sin.(2π .* y ./ unwrap(k.T)')]
    return elwise(k.k, px, py)
end
elwise(k::PeriodicKernel, x) = elwise(k, x, x)

# This gives us pretty-printing but is wrong in some case e.g. if k.T is a scalar
# `(ConstantKernel() ↻ 3) != PeriodicKernel(3, ConstantKernel())`
Base.show(io::IO, k::PeriodicKernel) = print(io, "(", k.k, " ↻ ", k.T, ")")

isMulti(k::PeriodicKernel) = isMulti(k.k)

"""
    k::Kernel ↻ l -> PeriodicKernel
    periodicise(k::Kernel, l) -> PeriodicKernel

Turn kernel `k` into a [`PeriodicKernel`](@ref) of period `l`.
"""
periodicise(k::Kernel, l) = PeriodicKernel(isconstrained(l) ? l : Positive(l), k)
const ↻ = periodicise

"""
    SpecifiedQuantityKernel <: Kernel
    SpecifiedQuantityKernel(col::Fixed, k::Kernel) -> SpecifiedQuantityKernel
	k::Kernel ← col::Fixed -> SpecifiedQuantityKernel

A kernel `k` that acts on the column(s) `col` of an input. Allows for input selection.

See also: [`takes_in`](@ref)
"""
mutable struct SpecifiedQuantityKernel <: Kernel
    col::Fixed
    k::Kernel
end

"""
	k::Kernel ← col::Union{Symbol, Fixed} -> SpecifiedQuantityKernel
	takes_in(k::Kernel, col::Union{Symbol, Fixed}) -> SpecifiedQuantityKernel

Specify that kernel `k` is a [`SpecifiedQuantityKernel`](@ref) which takes in data from the
input column(s) `col`.
"""
function takes_in(k::Kernel, col::Union{Symbol, Int, UnitRange{Int}, StepRange{Int, Int}})
    return SpecifiedQuantityKernel(Fixed(col), k)
end
takes_in(k::Kernel, col::Fixed) = SpecifiedQuantityKernel(col, k)
const ← = takes_in

@unionise function (k::SpecifiedQuantityKernel)(x::AbstractMatrix, y::AbstractMatrix)
    k.k(x[:, unwrap(k.col)], y[:, unwrap(k.col)])
end
@unionise (k::SpecifiedQuantityKernel)(x::AbstractMatrix) = k(x, x)

function elwise(k::SpecifiedQuantityKernel, x::AbstractMatrix)
    return [k(x[i, :]')[1] for i in 1:size(x, 1)] # Gotta keep the shape
end

@unionise function (k::SpecifiedQuantityKernel)(x::DataFrameRow, y::DataFrameRow)
    return k(DataFrame(x), DataFrame(y))
end
@unionise function (k::SpecifiedQuantityKernel)(x::AbstractDataFrame, y::DataFrameRow)
    return k(x, DataFrame(y))
end
@unionise function (k::SpecifiedQuantityKernel)(x::DataFrameRow, y::AbstractDataFrame)
    return k(DataFrame(x), y)
end
@unionise function (k::SpecifiedQuantityKernel)(x::AbstractDataFrame, y::AbstractDataFrame)
    # The tests still pass with this commented out, but I have the impression that it might
    # fail when we deal with real-world data, so I am leaving this here. The issue with
    # using it is that `disallowmissing` does not play nice with Nabla. So if we need to
    # come back to this, we'll need to implement it.
    # return k.k(disallowmissing(x[unwrap(k.col)]), disallowmissing(y[unwrap(k.col)]))
    if eltype(x[:, unwrap(k.col)]) <: AbstractVector
        return k.k(
            Matrix(hcat(x[:, unwrap(k.col)]...)'),
            Matrix(hcat(y[:, unwrap(k.col)]...)')
        )
    else
        return k.k(x[:, unwrap(k.col)], y[:, unwrap(k.col)])
    end
end
@unionise (k::SpecifiedQuantityKernel)(x::AbstractDataFrame) = k(x, x)
@unionise (k::SpecifiedQuantityKernel)(x::DataFrameRow) = k(x, x)

function elwise(k::SpecifiedQuantityKernel, x::AbstractDataFrame, y::AbstractDataFrame)
    if eltype(x[:, unwrap(k.col)]) <: AbstractVector
        return elwise(
            k.k,
            Matrix(hcat(x[:, unwrap(k.col)]...)'),
            Matrix(hcat(y[:, unwrap(k.col)]...)')
        )
    else
        return elwise(k.k, x[:, unwrap(k.col)], y[:, unwrap(k.col)])
    end
end
elwise(k::SpecifiedQuantityKernel, x::AbstractDataFrame) = elwise(k, x, x)
function elwise(k::SpecifiedQuantityKernel, x::DataFrameRow, y::DataFrameRow)
    return elwise(k, DataFrame(x), DataFrame(y))
end
function elwise(k::SpecifiedQuantityKernel, x::AbstractDataFrame, y::DataFrameRow)
    return elwise(k, x, DataFrame(y))
end
function elwise(k::SpecifiedQuantityKernel, x::DataFrameRow, y::AbstractDataFrame)
    return elwise(k, DataFrame(x), y)
end

Base.show(io::IO, k::SpecifiedQuantityKernel) = print(io, "(", k.k, " ← ", k.col, ")")
isMulti(k::SpecifiedQuantityKernel) = isMulti(k.k)

"""
    ConstantKernel <: Kernel

Kernel that returns 1.0 for every pair of points.
"""
struct ConstantKernel <: Kernel end
(k::ConstantKernel)(x, y) = ones(Float64, size(x, 1), size(y, 1))
(k::ConstantKernel)(x) = k(x, x)
function elwise(k::ConstantKernel, x, y)
    size(x) != size(y) && throw(DimensionMismatch("`x` and `y` must be of same size."))
    return ones(Float64, size(x, 1))
end
elwise(k::ConstantKernel, x) = ones(size(x, 1))
function Base.:+(k::Kernel, x)
    return isconstrained(x) ?
        SumKernel(k, x * ConstantKernel()) :
        (unwrap(x) ≈ zero(unwrap(x)) ? k : SumKernel(k, x * ConstantKernel()))
end
Base.:+(x, k::Kernel) = (+)(k::Kernel, x)
Base.convert(::Type{Kernel}, x::Real) = x ≈ 0.0 ? ZeroKernel() : Fixed(x) * ConstantKernel()
Base.show(io::IO, k::ConstantKernel) = print(io, "𝟏")

"""
    ZeroKernel <: Kernel

Zero kernel. Returns zero.
"""
struct ZeroKernel <: Kernel; end
(::ZeroKernel)(x, y) = zeros(size(x, 1), size(y, 1))
(k::ZeroKernel)(x) = k(x, x)
function elwise(::ZeroKernel, x, y)
    size(x) != size(y) && throw(DimensionMismatch("`x` and `y` must be of same size."))
    zeros(size(x, 1))
end
elwise(k::ZeroKernel, x) = zeros(size(x, 1))
Base.:+(k::Kernel, z::ZeroKernel) = k
Base.:+(z::ZeroKernel, k::Kernel) = k + z
Base.:+(z::ZeroKernel, k::ZeroKernel) = z
Base.:+(z::ZeroKernel, k::ScaledKernel) = k
Base.:+(k::ScaledKernel, z::ZeroKernel) = z + k
Base.:*(k::Kernel, z::ZeroKernel) = z
Base.:*(z::ZeroKernel, k::Kernel) = k * z
Base.:*(z::ZeroKernel, k::ZeroKernel) = z
Base.:*(z::ZeroKernel, k::ScaledKernel) = z
Base.:*(k::ScaledKernel, z::ZeroKernel) = z * k
Base.:*(x, z::ZeroKernel) = z
Base.:*(z::ZeroKernel, x) = x * z
Base.show(io::IO, z::ZeroKernel) = print(io, "𝟎")

"""
    DiagonalKernel <: Kernel

Diagonal kernel. Has unitary variance.
"""
struct DiagonalKernel <: Kernel end
function (::DiagonalKernel)(x, y)
    return float.(isapprox.(sq_pairwise_dist(x, y), 0))
end
(k::GPForecasting.DiagonalKernel)(x::DataFrame, y::DataFrame) = k(Matrix(x), Matrix(y))
function (k::DiagonalKernel)(x::DataFrameRow, y::DataFrameRow)
    return k(DataFrame(x), DataFrame(y))
end
function (k::DiagonalKernel)(x::AbstractDataFrame, y::DataFrameRow)
    return k(x, DataFrame(y))
end
function (k::DiagonalKernel)(x::DataFrameRow, y::AbstractDataFrame)
    return k(DataFrame(x), y)
end
(k::DiagonalKernel)(x) = k(x, x)

function elwise(k::DiagonalKernel, x, y)
    size(x) != size(y) && throw(DimensionMismatch("`x` and `y` must be of same size."))
    return float.([float.(x[i, :]) ≈ float.(y[i, :]) for i in 1:size(x, 1)])
end
elwise(k::DiagonalKernel, x::DataFrame, y::DataFrame) = elwise(k, Matrix(x), Matrix(y))
function elwise(k::DiagonalKernel, x::DataFrameRow, y::DataFrameRow)
    return elwise(k, DataFrame(x), DataFrame(y))
end
function elwise(k::DiagonalKernel, x::AbstractDataFrame, y::DataFrameRow)
    return elwise(k, x, DataFrame(y))
end
function elwise(k::DiagonalKernel, x::DataFrameRow, y::AbstractDataFrame)
    return elwise(k, DataFrame(x), y)
end
elwise(k::DiagonalKernel, x::Number, y) = elwise(k, [x], y)
elwise(k::DiagonalKernel, x, y::Number) = elwise(k, x, [y])
elwise(k::DiagonalKernel, x::Number, y::Number) = elwise(k, [x], [y])
elwise(k::DiagonalKernel, x) = ones(size(x, 1))
Base.show(io::IO, k::DiagonalKernel) = print(io, "δₓ")

"""
    DotKernel <: Kernel
    DotKernel(offset=Fixed(0.0)) -> DotKernel

Dot product kernel. Non-stationary.
"""
struct DotKernel <: Kernel
    offset

    DotKernel(offset=Fixed(0.0)) = new(offset)
end
@unionise function (k::DotKernel)(x::AbstractArray{<:Real}, y::AbstractArray{<:Real})
    return (x .- unwrap(k.offset)') * (y .- unwrap(k.offset)')'
end
(k::DotKernel)(x::Number, y) = k([x], y)
(k::DotKernel)(x, y::Number) = k(x, [y])
(k::DotKernel)(x::Number, y::Number) = k([x], [y])
@unionise (k::DotKernel)(x::AbstractArray{<:Real}) = k(x, x)

@unionise function elwise(k::DotKernel, x::AbstractArray{<:Real}, y::AbstractArray{<:Real})
    return reshape(sum(x .* y, dims=2), size(x, 1))
end
elwise(k::DotKernel, x::Number, y) = elwise(k, [x], y)
elwise(k::DotKernel, x, y::Number) = elwise(k, x, [y])
elwise(k::DotKernel, x::Number, y::Number) = elwise(k, [x], [y])
@unionise elwise(k::DotKernel, x::AbstractArray{<:Real}) = elwise(k, x, x)
function Base.show(io::IO, k::DotKernel)
    if k.offset ≈ Fixed(0.0)
        print(io, "<., .>")
    else
        print(io, "<. - ", k.offset, ". - ", k.offset, ">")
    end
end

"""
    HazardKernel <: Kernel
    HazardKernel(bias, scale) -> HazardKernel

Kernel tailor-made for hazards. It uses an augmented version of the [`DotKernel`](@ref).

# Fields
- `bias=Fixed(0.0)`: determines the projection of every hazard vector into the non-hazard
  space.
- `scale=Fixed(-1.0)`: represents a (possibly non-uniform) scale to be applied to every hazard
  vector `v` as `v .* scale`.

!!! note
    In order for a `HazardKernel` to make sense, we must have `abs(bias) < 1` and
    `size(unwrap(scale), 1) == 1` i.e. `scale` must be a row vector when unwrapped.
    It probably makes more sense to use normalised hazard vectors.
    Also, this is thought of as a "multiplicative" kernel.
"""
mutable struct HazardKernel <: Kernel
    bias
    scale

    function HazardKernel(bias=Fixed(0.0), scale=Fixed(-1.0))
        unwrap(bias) >= 1.0 && warn(LOGGER,
            """
            A HazardKernel with bias larger than or equal to 1.0 can yield unexpected
            results. Value received: $(unwrap(bias)).
            """
        )
        size(unwrap(scale), 1) != 1 && throw(ArgumentError("Scale must be a `RowVector`"))
        return new(bias, scale)
    end
end
@unionise function (k::HazardKernel)(x::AbstractArray{<:Real}, y::AbstractArray{<:Real})
    # First, augment the input space
    xl = [x[i, :] for i in 1:size(x, 1)]
    yl = [y[i, :] for i in 1:size(y, 1)]
    mask_x = isapprox.(xl, fill(zero(xl[1]), size(xl, 1))) # find which points have no hazard
    mask_y = isapprox.(yl, fill(zero(yl[1]), size(yl, 1)))
    h_x = .!mask_x .* fill(unwrap(k.bias), size(mask_x, 1)) # bias vector
    h_y = .!mask_y .* fill(unwrap(k.bias), size(mask_y, 1))
    scale = unwrap(k.scale) == -1 ? ones(1, size(x, 2)) : unwrap(k.scale)
    x_aug = hcat(x .* scale, (h_x + mask_x)) # scaling at the same time
    y_aug = hcat(y .* scale, (h_y + mask_y))

    return DotKernel()(x_aug, y_aug)
end
@unionise (k::HazardKernel)(x::AbstractArray{<:Real}) = k(x, x)
(k::HazardKernel)(x::Number, y) = k([x], y)
(k::HazardKernel)(x, y::Number) = k(x, [y])
(k::HazardKernel)(x::Number, y::Number) = k([x], [y])
@unionise function elwise(k::HazardKernel, x::AbstractArray{<:Real}, y::AbstractArray{<:Real})
    # First, augment the input space
    xl = [x[i, :] for i in 1:size(x, 1)]
    yl = [y[i, :] for i in 1:size(y, 1)]
    mask_x = isapprox.(xl, fill(zero(xl[1]), size(xl, 1))) # find which points have no hazard
    mask_y = isapprox.(yl, fill(zero(yl[1]), size(yl, 1)))
    h_x = .!mask_x .* fill(unwrap(k.bias), size(mask_x, 1)) # bias vector
    h_y = .!mask_y .* fill(unwrap(k.bias), size(mask_y, 1))
    scale = unwrap(k.scale) == -1 ? ones(1, size(x, 2)) : unwrap(k.scale)
    x_aug = hcat(x .* scale, (h_x + mask_x)) # scaling at the same time
    y_aug = hcat(y .* scale, (h_y + mask_y))

    return elwise(DotKernel(), x_aug, y_aug)
end
@unionise elwise(k::HazardKernel, x::AbstractArray{<:Real}) = elwise(k, x, x)
elwise(k::HazardKernel, x::Number, y) = elwise(k, [x], y)
elwise(k::HazardKernel, x, y::Number) = elwise(k, x, [y])
elwise(k::HazardKernel, x::Number, y::Number) = elwise(k, [x], [y])

"""
    ManifoldKernel <: Kernel
    ManifoldKernel(k::Kernel, nn::GPFNN) -> ManifoldKernel

Build a kernel under the Manifold GPs framework.

The inputs to the kernel `k` are pre-transformed by the neural network `nn`.
For more details, see:
    ["Manifold Gaussian Processes for Regression"](https://arxiv.org/abs/1402.5876)

# Fields
- `k::Kernel`: any regular kernel
- `nn::GPFNN`: a neural network.
"""
struct ManifoldKernel <: Kernel
    k::Kernel
    NN::GPFNN
end
function (k::ManifoldKernel)(x, y)
    return k.k(
        vcat((k.NN(x[i, :])' for i in 1:size(x, 1))...),
        vcat((k.NN(y[i, :])' for i in 1:size(y, 1))...)
    )
end
function (k::ManifoldKernel)(x::T, y::P) where {T <: Input, P <: Input}
    return k.k(
        T(vcat((k.NN(x.val[i, :])' for i in 1:size(x.val, 1))...)),
        P(vcat((k.NN(y.val[i, :])' for i in 1:size(y.val, 1))...))
    )
end
(k::ManifoldKernel)(x) = k.k(vcat((k.NN(x[i, :])' for i in 1:size(x, 1))...))
function (k::ManifoldKernel)(x::T) where T <: Input
    return k.k(T(vcat((k.NN(x.val[i, :])' for i in 1:size(x.val, 1))...)))
end
function elwise(k::ManifoldKernel, x, y)
    return elwise(
        k.k,
        vcat((k.NN(x[i, :])' for i in 1:size(x, 1))...),
        vcat((k.NN(y[i, :])' for i in 1:size(y, 1))...)
    )
end
function elwise(k::ManifoldKernel, x::T, y::P) where {T <: Input, P <: Input}
    return elwise(
        k.k,
        T(vcat((k.NN(x.val[i, :])' for i in 1:size(x.val, 1))...)),
        P(vcat((k.NN(y.val[i, :])' for i in 1:size(y.val, 1))...))
    )
end
elwise(k::ManifoldKernel, x) = elwise(k.k, vcat((k.NN(x[i, :])' for i in 1:size(x, 1))...))
function elwise(k::ManifoldKernel, x::T) where T <: Input
    return elwise(k.k, T(vcat((k.NN(x.val[i, :])' for i in 1:size(x.val, 1))...)))
end

Base.zero(::Kernel) = ZeroKernel()
Base.zero(::Type{Kernel}) = ZeroKernel()

"""
    NKN <: Kernel

Neural Network Kernel. Especial kind of kernel designed for automatic kernel design
applications. Based on https://arxiv.org/abs/1806.04326.

* Fields:
- `base_kernels::Vector{<:Kernel}`: Base kernels to be combined. Can be any valid `Kernel`
and have tunable parameters.
- `layers::Tuple{Vararg{Union{NNLayer, ProductLayer}}}`: Tuple containing `NNLayer` and 
`ProductLayer` objects which specify the NKN geometry.
"""
mutable struct NKN <: Kernel
    base_kernels::Vector{<:Kernel}
    layers::Tuple{Vararg{Union{NNLayer, ProductLayer}}}

    function NKN(base_kernels, layers)
        # bunch of checks
        if size(layers[1], 2) != length(base_kernels)
            throw(DimensionMismatch("""
                First layer has dimensions $(size(layers[1])), base kernels have length
                $(length(base_kernels)).
            """))
        end
        if size(layers[end], 1) != 1
            throw(DimensionMismatch("Output of last layer must be of length 1."))
        end
        mismatch = [size(layers[i], 1) != size(layers[i + 1], 2) for i in 1:length(layers)-1]
        if any(mismatch)
            throw(DimensionMismatch(
                """
                Incompatible layer dimensions between layers $(findall(mismatch)) and their
                following layers.
                """
            ))
        end
        return new(base_kernels, layers)
    end
end

# Some convenient constructors to help with the initialisation of NKNs
function LinearLayer(s::Tuple{Int, Int})
    return NNLayer(
        # Initialisation heuristics stolen directly from the original implementation:
        # https://github.com/ssydasheng/GPflow-Slim/blob/223b9e30e8a7969496a3a53acd3a39835f5a3f8b/gpflowSlim/neural_kernel_network/neural_kernel_network_wrapper.py#L99
        Positive(rand(Uniform(1.0 / (2 * s[2]), 3.0 / (2 * s[2])), s...)),
        Positive(0.01 * ones(s[1])),
        Fixed(x -> x) # no non-linearity
    )
end

function ProductLayer(s::Tuple{Int, Int})
    C = rand(Bool, s...)
    # Here we avoid entire lines of `false`s. Not very efficient, but this
    # constructor will only be called once every run.
    # We still might have repeated lines, which is not ideal.
    for i in 1:s[1]
        if sum(C[i, :]) == 0
            C[i, rand(1:s[2])] = true
        end
    end
    return ProductLayer(Fixed(C))
end

# One has to be careful here when setting `unique = true`, as this will loop indefinitely
# in case s[1] is larger than the total number of unique permutations of the seed.
# That is s[2]! / (step! (s[2] - step)!). The check is not executed here because factorials
# blow up really fast.
function ProductLayer(s::Tuple{Int, Int}, step::Int; unique=false)
    if step > s[2] || step < 2
        throw(ArgumentError("step must be > 2 and <= $(s[2])"))
    end
    seed = Vector{Bool}(vcat(fill(true, step), fill(false, s[2] - step)))
    all_rows = Vector{Adjoint{Bool,Array{Bool,1}}}()
    for i in 1:s[1]
        if unique
            r = shuffle(seed)
            if !in(r, all_rows)
                push!(all_rows, r')
            else
                i -= 1
            end
        else
            push!(all_rows, shuffle(seed)')
        end
    end
    return ProductLayer(Fixed(vcat(all_rows...)))
end

# hadprod(v) = reduce((x, y) -> x .* y, [v[i] for i in 1:length(v)])
hadprod(v) = reduce((x, y) -> x .* y, v)

function (k::NKN)(x, y)
    # The type declaration here is because of Nabla, otherwise it breaks on the backward
    # pass.
    covs::AbstractArray{Any} = [bk(x, y) for bk in k.base_kernels]
    for layer in k.layers
        if layer isa NNLayer
            W = unwrap(layer.W)
            b = unwrap(layer.b)
            c = covs
            # This reshaping is also because of Nabla. It turns a Nabla-wrapped array into
            # an array of Nabla-wrapped objects, which allow proper dispatching.
            covs = reshape([W[i] for i = 1:length(W)], size(W)...) * c +
                reshape([b[i] for i = 1:length(b)], size(b)...) .*
                fill(ones(size(x, 1), size(y, 1)), length(b))
        else
            C = unwrap(layer.C)
            covs = map([covs[C[i, :]] for i in 1:size(C, 1)]) do c
                if isempty(c)
                    # This is necessary for the case in which we have an entire line of
                    # `false`s. In that case, we are not combining any of the kernels, so
                    # we default the result to the equivalent of a ConstantKernel.
                    # Ideally one wouldn't feed lines like this to the kernel, but it may
                    # happen when generating those randomly.
                    return Ones(size(covs[1]))
                else
                    return hadprod(c)
                end
            end
        end
    end
    length(covs) > 1 && throw(
        error("Final output has length $(length(covs)), something went wrong.")
    )
    return covs[1]
end

(k::NKN)(x) = k(x, x)
Base.show(io::IO, k::NKN) = show(io, equivalent_kernel(k))

"""
    equivalent_kernel(k::NKN)

Obtain the `Kernel` that corresponds to the current specific state of a `NKN`, `k`.
"""
function equivalent_kernel(k::NKN)
    ek::Vector{Kernel} = k.base_kernels
    for layer in k.layers
        if layer isa NNLayer
            W = unwrap(layer.W)
            b = unwrap(layer.b)
            ek = Vector{Kernel}(
                W * ek::Vector{Kernel} + b .* fill(ConstantKernel(), length(b))
            )
        else
            ek = p_layer(unwrap(layer.C), ek)
        end
    end
    length(ek) > 1 && throw(
        error("Final output has length $(length(ek)), something went wrong.")
    )
    return ek[1]
end
function p_layer(cf::Matrix{Bool}, ek::Vector{<:Kernel})::Vector{Kernel}
    return Kernel[prod(ek[cf[i, :]]) for i in 1:size(cf, 1)]
end

"""
    SparseKernel{K <: Kernel} <: Kernel

Not supposed to be used directly by the user. This is automatically called under the hood,
whenever necessary.
"""
mutable struct SparseKernel{K <: Kernel} <: Kernel
    k::K
    Xm
    n
    σ²
end
SparseKernel(k::Kernel, Xm, σ²) = SparseKernel(k, Xm, Fixed(size(unwrap(Xm), 1)), σ²)
(k::SparseKernel)(x) = k.k(x, unwrap(k.Xm))
(k::SparseKernel)() = k.k(unwrap(k.Xm))
Base.show(io::IO, k::SparseKernel) = print(io, "Sparse(", k.k, ")")

"""
    CosineKernel <: Kernel

Kernel that computes cos(|x - y|). Allows for negative correlations.
"""
struct CosineKernel <: Kernel end
(::CosineKernel)(x, y) = cos.(pairwise_dist(x, y))
(k::CosineKernel)(x) = k(x, x)
elwise(::CosineKernel, x, y) = cos.(elwise_dist(x, y))
elwise(::CosineKernel, x) = ones(size(x, 1))

"""
    HeteroskedasticDiagonalKernel <: Kernel

Diagonal kernel which has different variances for each input location. *Only accepts integer
inputs in the interval [0, N]*. All variances in [0, N] must be specified.
"""
struct HeteroskedasticDiagonalKernel <: Kernel
    vars

    function HeteroskedasticDiagonalKernel(vars)
        vs = isconstrained(vars) ? vars : Positive(vars)
        return new(vs)
    end
end
HeteroskedasticDiagonalKernel(n::Int) = HeteroskedasticDiagonalKernel(ones(n))
# Gradiente aqui nao faz sentido...
function (k::HeteroskedasticDiagonalKernel)(x::ArrayOrReal, y::ArrayOrReal)
    K = DiagonalKernel()(x, y)
    n = length(unwrap(k.vars)) - 1
    hodx = Int.(x)
    hody = Int.(y)
    if any(0 .> hodx) || any(n .< hodx) || any(0 .> hody) || any(n .< hody)
        throw(ArgumentError(
            "Kernel $k only takes inputs between 0 and $n."
        ))
    end
    vx = unwrap(k.vars)[hodx .+ 1]
    vy = unwrap(k.vars)[hody .+ 1]
    covs = vx * vy' # Outer product
    # Multiplying by the DiagonalKernel eliminates all cross-products
    return covs .* K
end
(k::HeteroskedasticDiagonalKernel)(x::ArrayOrReal) = k(x, x)

Base.one(::Kernel) = ConstantKernel()
Base.one(::Type{GPForecasting.Kernel}) = ConstantKernel()
