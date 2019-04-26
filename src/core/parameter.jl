isconstrained(x) = false

# Base cases of parameter recursions:
unwrap(x) = x
name(x) = nothing

"""
    pack(k::Number) -> Vector

Pack number inside a vector.
"""
pack(x::Number) = [x]

"""
    unpack(x::Number, y::Vector) -> Number

Unpack number from inside a vector.
"""
@unionise unpack(x::Number, y::Vector) = y[1]
pack(x::AbstractArray) = x[:]
@unionise unpack(x::AbstractArray, y::Vector) = reshape(y, size(x)...)
set(x::Number, y::Number) = y
set(x::AbstractArray, y::AbstractArray) = y

"""
    Parameter

Abstract type for all types of paramaters.
"""
abstract type Parameter end
# The following two lines assert that each subtype of parameter has its wrapped value as the
# first field. We could do something similar to the kernels to make this fully flexible,
# but I think this suffices.
parameter(p::Parameter) = getfield(p, 1)
others(p::Parameter) = collect(Iterators.rest(getfields(p), 1))

# Default parameter behaviour:
reconstruct(p::Parameter, parameter, others) = typeof(p)(parameter, others...)
unwrap(p::Parameter) = unwrap(parameter(p))
name(p::Parameter) = name(parameter(p))
Base.show(io::IO, p::Parameter) = show(IOContext(io, :compact => true), parameter(p))

"""
    pack(k::Parameter) -> Vector

Pack parameter in a vector that can be optimised.
Automatically converts the parameter to the space in which it should be optimised.
"""
pack(p::Parameter) = pack(parameter(p))
@unionise unpack(x::T, y::T) where T <: Parameter = y

"""
    unpack(k::Parameter, y) -> Parameter

Unpack a vector value into a `Parameter`.
Used for updating parameter values.
"""
@unionise unpack(x::Parameter, y::Vector) =
    reconstruct(x, unpack(parameter(x), y), others(x))
set(x::Parameter, y) = reconstruct(x, set(parameter(x), y), others(x))

# Fallback when x and y have different types
Base.isapprox(x::Parameter, y::Parameter) = false

function Base.isapprox(x::T, y::T) where T<:Parameter
    return all(i -> _isapprox(getfield(x, i), getfield(y, i)), 1:fieldcount(T))
end

_isapprox(a, b) = isapprox(a, b)
_isapprox(a::AbstractString, b::AbstractString) = a == b

"""
    isconstrained(x::Parameter)

Return `true` if the parameter is subject to any form of numerical constraint, `false`
otherwise.
"""
isconstrained(x::Parameter) = isconstrained(x.p)


"""
   Fixed{T} <: Parameter

Type for parameters that should not be optimised.
"""
mutable struct Fixed{T} <: Parameter
    p::T
end
pack(x::Fixed) = Float64[]
@unionise unpack(x::Fixed, y::Vector) = x
isconstrained(x::Fixed) = true

"""
    Positive{T} where T <: Parameter

Type for parameters that should be kept positive during the optimisation process. Will be
transformed to the log space before optimisation.
"""
mutable struct Positive{T} <: Parameter
    p::T
    ε::Float64
end
Positive(p) = Positive(p, _EPSILON_)
function Positive{T}(p::T) where T
    return Positive{T}(p, _EPSILON_)
end
pack(x::Positive) = map(y -> log.(max.(y .- x.ε, x.ε)), pack(x.p))
@unionise function unpack(x::Positive, y::Vector)
    return Positive(unpack(x.p, map(z -> exp.(z) .+ x.ε, y)), x.ε)
end
isconstrained(x::Positive) = true

"""
    Bounded{T} <: Parameter

Type for parameters that should be kept within `lb` and `ub` during the optimisation process.
Optional `ε` is used to map [lb, ub] to [lb + ε, ub - ε] for numerical stability.
"""
mutable struct Bounded{T} <: Parameter
    p::T
    lb
    ub
    ε::Float64
end
Bounded(p, lb, ub) = Bounded(p, lb, ub, _EPSILON_)
function Bounded{T}(p::T, lb, ub) where T
   return Bounded{T}(p, lb, ub, _EPSILON_)
end
Bounded(p, ub) = Bounded(p, 0.0, ub, _EPSILON_)
function Bounded{T}(p::T, ub) where T
   return Bounded{T}(p, 0.0, ub, _EPSILON_)
end
clip(x, lb, ub) = min.(max.(x, lb), ub)
function bounded2R(x, lb, ub, ε)
    # Map x from [lb, ub] to [lb + ε, ub - ε].
    x = (x .- lb) ./ (ub .- lb) .* (ub .- lb .- 2 .* ε) .+ lb .+ ε
    # Clip in case x isn't in [lb + ε, ub - ε].
    x = clip(x, lb .+ ε, ub .- ε)
    # Map x from [lb + ε, ub - ε] to [log(ε) - log(ub - lb), -log(ε) + log(ub - lb)].
    return log.(x .- lb) .- log.(ub .- x)
end
function R2bounded(x, lb, ub, ε)
    # Map x from [log(ε) - log(ub - lb), -log(ε) + log(ub - lb)] to [lb + ε, ub - ε].
    x = (ub .+ lb .* exp.(-x)) ./ (one.(x) .+ exp.(-x))
    # Map x from [lb + ε, ub - ε] to [lb, ub].
    return (x .- lb .- ε) ./ (ub .- lb .- 2 .* ε) .* (ub .- lb) .+ lb
end
function pack(x::Bounded)
    lb = unwrap(x.lb)
    ub = unwrap(x.ub)
    return bounded2R(pack(x.p), lb, ub, x.ε)
end
@unionise function unpack(x::Bounded, y::Vector)
    lb = unwrap(x.lb)
    ub = unwrap(x.ub)
    return Bounded(unpack(x.p, R2bounded(y, lb, ub, x.ε)), lb, ub, x.ε)
end
isconstrained(x::Bounded) = true


"""
    DynamicBound

Type for allowing moving bounds for `Bounded` types. Calls a function `f` over the unwrapped
arguments `args`.
"""
mutable struct DynamicBound
    f::Function
    args::Vector
end
unwrap(db::DynamicBound) = db.f(unwrap.(db.args)...)

"""
    Named{T} <: Parameter

Type for parameters that can be identified by a `name`.
"""
mutable struct Named{T} <: Parameter
    p::T
    name::String
end
name(x::Named) = x.name
