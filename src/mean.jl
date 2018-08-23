export Mean, PosteriorMean, ConstantMean, FunctionMean, FunctionalMean, ZeroMean, ScaledMean

# Eventually could make a macro @symmetrize which symmetrizes operations
using Nabla
using Missings
import Base: *, +, /, -, \, ^

type Operation
    f::Fixed
    Operation(f::Function) = new(Fixed(f))
end
(*)(m...) = Operation(*)(m...)
(/)(m...) = Operation(/)(m...)
(+)(m...) = Operation(+)(m...)
(-)(m...) = Operation(-)(m...)
(^)(m...) = (isa(m[2], Real) && m[2] ≈ 0) ? ConstantMean() : Operation(^)(m[1], m[2])
# Special cases for exponentiation: Convert integers to floats.
(^)(n::Integer, m̄::Mean) = (^)(Float64(n), m̄::Mean)
(^)(m̄::Mean, n::Integer) = (^)(m̄::Mean, Float64(n))

"""
    FunctionMean <: Mean

Mean that follows function `m`.
"""
type FunctionMean <: Mean
    m::Fixed
    FunctionMean(m::Union{Fixed, Function}) = isa(m, GPForecasting.Fixed) ? new(m) : new(Fixed(m))
end
(m::FunctionMean)(x) = broadcast(unwrap(m.m), x)

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
type PosteriorMean <: Mean
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

"""
    ConstantMean <: Mean

The most primitive building block of a Mean.
"""
type ConstantMean <: Mean
    ConstantMean() = new()
    ConstantMean(x) = unwrap(x) == 0 ? ZeroMean() : ScaledMean(x, ConstantMean())
end
(m::ConstantMean)(x) = ones(Float64, size(x, 1))


"""
    ScaledMean <: Mean

Corresponds to a primitive mean `\mu` multiplied by a `scale`
"""
type ScaledMean <: Mean
    scale
    μ::Mean
    ScaledMean(s) = ScaledMean(s, ConstantMean())
    ScaledMean(s, m) = unwrap(s) == 0 ? ZeroMean() : new(s, m)
end
ScaledMean() = ScaledMean(1.0)
function show(io::IO, m::ScaledMean)
    return isa(m.μ, ConstantMean) ? print(io, "$(m.scale)") : print(io, "$(m.scale) * $(m.μ)")
end
(m::ScaledMean)(x) = unwrap(m.scale) .* m.μ(x)

"""
    ZeroMean <: Mean

Zero mean. Returns zero.
"""
type ZeroMean <: Mean; end
(::ZeroMean)(x) = zeros(size(x, 1))
show(io::IO, k::ZeroMean) = print(io, "𝟎")
(f::Operation)(z::ZeroMean, z̄::ZeroMean) = isa(unwrap(f.f), typeof(/)) ? 0/0 : z # Return default julia behaviour if division
(f::Operation)(z::Union{Parameter, Real}, z̄::ZeroMean) = (f)(ConstantMean(z), z̄)
(f::Operation)(z::ZeroMean, z̄::Union{Parameter, Real}) = (f)(z, ConstantMean(z̄))
function (f::Operation)(z::ZeroMean, m::Mean)
    if (unwrap(f.f) == *) return z
    elseif (unwrap(f.f) == /) return z
    elseif (unwrap(f.f) == ^) return z
    elseif (unwrap(f.f) == +)  return m
    elseif (unwrap(f.f) == -)  return -m end
end
function (f::Operation)(m::Mean, z::ZeroMean)
    if (unwrap(f.f) == *) return f(z, m)
    elseif (unwrap(f.f) == /) return ConstantMean(Inf) # Depends on what behaviour we want for this.
    elseif (unwrap(f.f) == ^) return ConstantMean(1.0)
    elseif (unwrap(f.f) == +)  return f(z, m)
    elseif (unwrap(f.f) == -)  return m end
end

"""
    VariableMean <: Mean

A mean that cannot be reduced to a ConstantMean.
"""
type VariableMean <: Mean
    μ::Union{Mean, Parameter, Real}
    μ̄::Union{Mean, Parameter, Real}
    f::Fixed
end
show(io::IO, m::VariableMean) = print(io, "$(m.μ) $(unwrap(m.f)) $(m.μ̄)")
(m::VariableMean)(x) = broadcast(unwrap(m.f), m.μ(x), m.μ̄(x))
(f::Operation)(m::Mean, m̄::Mean) = VariableMean(m, m̄, f.f)
(f::Operation)(m::Mean) = VariableMean(ConstantMean(unwrap(f.f)(1.0)), m, Fixed(*)) # Unitary operations on means
function (f::Operation)(m::Union{Real, Parameter}, m̄::Mean)
    return unwrap(m) == 0 ? (f)(ZeroMean(), m̄) : VariableMean(ConstantMean(m), m̄, f.f)
end
function (f::Operation)(m::Mean, m̄::Union{Real, Parameter})
    return unwrap(m̄) == 0 ? (f)(m, ZeroMean()) : VariableMean(m, ConstantMean(m̄), f.f)
end

zero(::Mean) = ConstantMean(0)
