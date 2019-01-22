export Mean, SumMean, ProductMeant, ConstantMean, FunctionMean, ZeroMean, ScaledMean,
PosteriorMean

(^)(m::Mean, n::Integer) = Base.power_by_squaring(m, n)

"""
    ScaledMean <: Mean

Result from the multiplication of a `Mean` by a number or `Parameter`.
"""
mutable struct ScaledMean <: Mean
    scale
    m::Mean
end
(m::ScaledMean)(x) = unwrap(m.scale) .* m.m(x)
function (*)(x, k::Mean)
    return isconstrained(x) ?
        ScaledMean(x, k) :
        (unwrap(x) ≈ zero(unwrap(x)) ? ZeroMean() : ScaledMean(x, k))
end
function (*)(x, k::ScaledMean)
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
(*)(k::Mean, x) = (*)(x, k::Mean)
function (+)(k1::Mean, k2::ScaledMean)
    return unwrap(k2.scale) ≈ zero(unwrap(k2.scale)) ? k1 : SumMean(k1, k2)
end
function (+)(k1::ScaledMean, k2::Mean)
    return unwrap(k1.scale) ≈ zero(unwrap(k1.scale)) ? k2 : SumMean(k1, k2)
end
function (+)(k1::ScaledMean, k2::ScaledMean)
    if unwrap(k1.scale) ≈ zero(unwrap(k1.scale))
        return k2
    elseif unwrap(k2.scale) ≈ zero(unwrap(k2.scale))
        return k1
    else
        return SumMean(k1, k2)
    end
end
show(io::IO, k::ScaledMean) = print(io, "($(k.scale) * $(k.m))")

"""
    SumMean <: Mean

Mean built by adding two Means, `m1` and `m2`.
"""
mutable struct SumMean <: Mean
    m1::Mean
    m2::Mean
end
(+)(k1::Mean, k2::Mean) = SumMean(k1, k2)
(k::SumMean)(x) = k.m1(x) .+ k.m2(x)
show(io::IO, k::SumMean) = print(io, "($(k.m1) + $(k.m2))")

"""
    ProductMean <: Mean

Mean built by multiplying two Means, `m1` and `m2`.
"""
mutable struct ProductMean <: Mean
    m1::Mean
    m2::Mean
end
(*)(k1::Mean, k2::Mean) = ProductMean(k1, k2)
function (*)(k1::Mean, k2::ScaledMean)
    return unwrap(k2.scale) ≈ zero(unwrap(k2.scale)) ? Mean(0) : ProductMean(k1, k2)
end
function (*)(k1::ScaledMean, k2::Mean)
    return unwrap(k1.scale) ≈ zero(unwrap(k1.scale)) ? Mean(0) : ProductMean(k1, k2)
end
function (*)(k1::ScaledMean, k2::ScaledMean)
    if unwrap(k1.scale) ≈ zero(unwrap(k1.scale))
        return Mean(0)
    elseif unwrap(k2.scale) ≈ zero(unwrap(k2.scale))
        return Mean(0)
    else
        return ProductMean(k1, k2)
    end
end
(k::ProductMean)(x) = k.m1(x) .* k.m2(x)
show(io::IO, k::ProductMean) = print(io, "($(k.m1) * $(k.m2))")

"""
    ConstantMean <: Mean

Mean that returns 1.0 for every point.
"""
struct ConstantMean <: Mean end
(k::ConstantMean)(x) = ones(Float64, size(x, 1))
(k::ConstantMean)(x::Vector{Input}) = ones(Float64, size(vcat([c.val for c in x]...), 1))
(k::ConstantMean)(x::Real, y::Real) = 1.0
function (+)(k::Mean, x)
    return isconstrained(x) ?
        SumMean(k, x * ConstantMean()) :
        (unwrap(x) ≈ zero(unwrap(x)) ? k : SumMean(k, x * ConstantMean()))
end
(+)(x, k::Mean) = (+)(k::Mean, x)
convert(::Type{Mean}, x::Real) = x ≈ 0.0 ? ZeroMean() : Fixed(x) * ConstantMean()
show(io::IO, k::ConstantMean) = print(io, "𝟏")
ConstantMean(x) = unwrap(x) == 0 ? ZeroMean() : x * ConstantMean()

"""
    ZeroMean <: Mean

Zero Mean. Returns zero.
"""
struct ZeroMean <: Mean; end
(::ZeroMean)(x) = zeros(size(x, 1))
(::ZeroMean)(x::Vector{Input}) = zeros(size(vcat([c.val for c in x]...), 1))
(+)(k::Mean, z::ZeroMean) = k
(+)(z::ZeroMean, k::Mean) = k + z
(+)(z::ZeroMean, k::ZeroMean) = z
(+)(z::ZeroMean, k::ScaledMean) = k
(+)(k::ScaledMean, z::ZeroMean) = z + k
(*)(k::Mean, z::ZeroMean) = z
(*)(z::ZeroMean, k::Mean) = k * z
(*)(z::ZeroMean, k::ZeroMean) = z
(*)(z::ZeroMean, k::ScaledMean) = z
(*)(k::ScaledMean, z::ZeroMean) = z * k
(*)(x, z::ZeroMean) = z
(*)(z::ZeroMean, x) = x * z
show(io::IO, z::ZeroMean) = print(io, "𝟎")

zero(::Mean) = ZeroMean()
zero(::Type{GPForecasting.Mean}) = ZeroMean()

"""
    FunctionMean <: Mean

Mean that follows function `m`.
"""
struct FunctionMean <: Mean
    m::Fixed
    FunctionMean(m::Union{Fixed, Function}) = isa(m, GPForecasting.Fixed) ? new(m) : new(Fixed(m))
end
(m::FunctionMean)(x) = broadcast(unwrap(m.m), x)
(m::FunctionMean)(x::Input) = broadcast(unwrap(m.m), x.val)
(m::FunctionMean)(x::Vector{Input}) = broadcast(unwrap(m.m), vcat([c.val for c in x]...))

"""
    PosteriorMean <: Mean

Posterior mean for a GP.

# Fields:
- `k`: prior kernel.
- `m`: prior mean.
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
show(io::IO, k::PosteriorMean) = print(io, "Posterior($(k.k), $(k.m))")
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
