Base.:^(m::Mean, n::Integer) = Base.power_by_squaring(m, n)

"""
    ScaledMean <: Mean
    ScaledMean(scale, mean::Mean) -> ScaledMean

Result from the multiplication of a `Mean` by a `Number` or `Parameter`.
"""
mutable struct ScaledMean <: Mean
    scale
    m::Mean
end
(m::ScaledMean)(x) = unwrap(m.scale) .* m.m(x)
function Base.:*(x, k::Mean)
    return isconstrained(x) ?
        ScaledMean(x, k) :
        (unwrap(x) ≈ zero(unwrap(x)) ? ZeroMean() : ScaledMean(x, k))
end
function Base.:*(x, k::ScaledMean)
    if isconstrained(x)
        return ScaledMean(x, k)
    else
        return unwrap(x) * unwrap(k.scale) ≈ zero(unwrap(x)) ? ZeroMean() :
            (
                isconstrained(k.scale) ?
                ScaledMean(x, k) :
                ScaledMean(unwrap(x) * unwrap(k.scale), k.m)
            )
    end
end
Base.:*(k::Mean, x) = (*)(x, k::Mean)
function Base.:+(k1::Mean, k2::ScaledMean)
    return unwrap(k2.scale) ≈ zero(unwrap(k2.scale)) ? k1 : SumMean(k1, k2)
end
function Base.:+(k1::ScaledMean, k2::Mean)
    return unwrap(k1.scale) ≈ zero(unwrap(k1.scale)) ? k2 : SumMean(k1, k2)
end
function Base.:+(k1::ScaledMean, k2::ScaledMean)
    if unwrap(k1.scale) ≈ zero(unwrap(k1.scale))
        return k2
    elseif unwrap(k2.scale) ≈ zero(unwrap(k2.scale))
        return k1
    else
        return SumMean(k1, k2)
    end
end
Base.show(io::IO, k::ScaledMean) = print(io, "(", k.scale, " * ", k.m, ")")

"""
    SumMean <: Mean
    SumMean(m1::Mean, m2::Mean) -> SumMean

Mean built by adding two Means, `m1` and `m2`.
"""
mutable struct SumMean <: Mean
    m1::Mean
    m2::Mean
end

Base.:+(k1::Mean, k2::Mean) = SumMean(k1, k2)

(k::SumMean)(x) = k.m1(x) .+ k.m2(x)

Base.show(io::IO, k::SumMean) = print(io, "(", k.m1, " + ", k.m2, ")")

"""
    ProductMean <: Mean
    ProductMean(m1::Mean, m2::Mean) -> ProductMean

Mean built by multiplying two Means, `m1` and `m2`.
"""
mutable struct ProductMean <: Mean
    m1::Mean
    m2::Mean
end

Base.:*(k1::Mean, k2::Mean) = ProductMean(k1, k2)
function Base.:*(k1::Mean, k2::ScaledMean)
    return unwrap(k2.scale) ≈ zero(unwrap(k2.scale)) ? zero(Mean) : ProductMean(k1, k2)
end
function Base.:*(k1::ScaledMean, k2::Mean)
    return unwrap(k1.scale) ≈ zero(unwrap(k1.scale)) ? zero(Mean) : ProductMean(k1, k2)
end
function Base.:*(k1::ScaledMean, k2::ScaledMean)
    if unwrap(k1.scale) ≈ zero(unwrap(k1.scale))
        return zero(Mean)
    elseif unwrap(k2.scale) ≈ zero(unwrap(k2.scale))
        return zero(Mean)
    else
        return ProductMean(k1, k2)
    end
end

(k::ProductMean)(x) = k.m1(x) .* k.m2(x)

Base.show(io::IO, k::ProductMean) = print(io, "(", k.m1, " * ", k.m2, ")")

"""
    ConstantMean <: Mean

Mean that returns 1.0 for every point.
"""
struct ConstantMean <: Mean end

(k::ConstantMean)(x) = ones(Float64, size(x, 1))
(k::ConstantMean)(x::Vector{Input}) = ones(Float64, size(vcat([c.val for c in x]...), 1))
(k::ConstantMean)(x::Real, y::Real) = 1.0

function Base.:+(k::Mean, x)
    isconstrained(x) && return SumMean(k, x * ConstantMean())
    return unwrap(x) ≈ zero(unwrap(x)) ? k : SumMean(k, x * ConstantMean())
end
Base.:+(x, k::Mean) = (+)(k::Mean, x)

Base.convert(::Type{Mean}, x::Real) = x ≈ 0.0 ? ZeroMean() : Fixed(x) * ConstantMean()

Base.show(io::IO, ::ConstantMean) = print(io, "𝟏")

# TODO: remove this, because all constructors should be such that `T(args...) isa T`
# replace with e.g. rename `constant_mean` and `@deprecate ConstantMean(x) constant_mean(x)`
ConstantMean(x) = unwrap(x) == 0 ? ZeroMean() : x * ConstantMean()

"""
    ZeroMean <: Mean

Zero Mean. Returns zero.
"""
struct ZeroMean <: Mean; end
(::ZeroMean)(x) = zeros(size(x, 1))
(::ZeroMean)(x::Vector{Input}) = zeros(size(vcat([c.val for c in x]...), 1))

Base.:+(k::Mean, z::ZeroMean) = k
Base.:+(z::ZeroMean, k::Mean) = k + z
Base.:+(z::ZeroMean, k::ZeroMean) = z
Base.:+(z::ZeroMean, k::ScaledMean) = k
Base.:+(k::ScaledMean, z::ZeroMean) = z + k

Base.:*(k::Mean, z::ZeroMean) = z
Base.:*(z::ZeroMean, k::Mean) = k * z
Base.:*(z::ZeroMean, k::ZeroMean) = z
Base.:*(z::ZeroMean, k::ScaledMean) = z
Base.:*(k::ScaledMean, z::ZeroMean) = z * k
Base.:*(x, z::ZeroMean) = z
Base.:*(z::ZeroMean, x) = x * z

Base.show(io::IO, ::ZeroMean) = print(io, "𝟎")

Base.zero(::Mean) = ZeroMean()
Base.zero(::Type{GPForecasting.Mean}) = ZeroMean()

"""
    FunctionMean <: Mean
    FunctionMean(m::Union{Function, Fixed}) -> FunctionMean

Mean that follows function `m`.
"""
struct FunctionMean <: Mean
    m::Fixed
end
FunctionMean(m::Function) = FunctionMean(Fixed(m))

(m::FunctionMean)(x) = broadcast(unwrap(m.m), x)
(m::FunctionMean)(x::Input) = broadcast(unwrap(m.m), x.val)
(m::FunctionMean)(x::Vector{Input}) = broadcast(unwrap(m.m), vcat([c.val for c in x]...))

"""
    PosteriorMean <: Mean

Posterior mean for a GP.

# Fields:
- `k::Kernel`: prior kernel.
- `m::Mean`: prior mean.
- `x`: points over which `m` was conditioned.
- `U`: Cholesky decomposition of the covariance matrix.
- `y`: values corresponding to the points `x`.
"""
struct PosteriorMean <: Mean
    k::Kernel
    m::Mean
    x
    U
    y

    PosteriorMean(k, m, x, U, y) = new(k, m, Fixed(x), Fixed(U), Fixed(y))
end

Base.show(io::IO, k::PosteriorMean) = print(io, "Posterior(", k.k, k.m, ")")

function (m::PosteriorMean)(x)
    xd = unwrap(m.x)
    U = unwrap(m.U)
    yd = unwrap(m.y)
    means = m.m(x)
    meansv = stack([means[:, i] for i in 1:size(means, 2)])
    ymm = yd .- m.m(xd)
    ydv = stack([ymm[:, i] for i in 1:size(ymm, 2)])
    posmeans = meansv .+ (m.k(x, xd) / U) * (U' \ ydv)
    return unstack(posmeans, size(means, 2))
end

"""
    TitsiasPosteriorMean <: Mean
    TitsiasPosteriorMean(k::Kernel, m::Mean, x, Xm, Uz, σ², y) -> TitsiasPosteriorMean

Posterior mean for a sparse GP under Titsias' approximation.

See: ["Variational Learning of Inducing Variables in Sparse Gaussian Processes"](http://proceedings.mlr.press/v5/titsias09a.html)
"""
mutable struct TitsiasPosteriorMean <: Mean
    k::Kernel
    m::Mean
    x
    Xm
    Uz
    σ²
    y
    function TitsiasPosteriorMean(k, m, x, Xm, Uz, σ², y)
        return new(
            k,
            m,
            Fixed(x),
            Fixed(Xm),
            Fixed(Uz),
            Fixed(σ²),
            Fixed(y)
        )
    end
end
function _titsposmean(m::TitsiasPosteriorMean, x)
    xn = unwrap(m.x)
    Xm = unwrap(m.Xm)
    Uz = unwrap(m.Uz)
    σ² = unwrap(m.σ²)
    yd = unwrap(m.y)

    ymm = yd .- m.m(xn)
    ydv = stack([ymm[:, i] for i in 1:size(ymm, 2)])
    Kxm = m.k(x, Xm)
    Kmn = m.k(Xm, xn)
    posmeans = (1/σ²) .* ((Kxm / Uz) * (Uz' \ Kmn)) * ydv
    return unstack(posmeans, size(yd, 2))
end
function (m::TitsiasPosteriorMean)(x)
    if is_not_noisy(m.k)
        notice(LOGGER,
            """
            Working on the extended input space. Output will be two dimensional,
            corresponding to the noisy and denoised predictions. To compute only the
            noisy (denoised) predictions, please wrap your input in `Observed` (`Latent`).
            """
        )
        val = _titsposmean(m::TitsiasPosteriorMean, x)
        return [val val]
    else
        return _titsposmean(m::TitsiasPosteriorMean, x)
    end
end
function (m::TitsiasPosteriorMean)(x::Input)
    xx = is_not_noisy(m.k) ? x.val : x
    return _titsposmean(m::TitsiasPosteriorMean, xx)
end
