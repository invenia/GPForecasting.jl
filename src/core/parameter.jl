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
    pack(df::DataFrame) -> Vector

Pack DataFrame in a vector that can be optimised. This becomes necessary when we want to
optimise the inducing points for sparse GPs with multi-variate inputs.
"""
pack(df::DataFrame) = vec(Matrix(df))

# Here's a bunch of things that may or not not be useful for Nabla. On one hand, they act
# on common types, but on the other hand, this is a very specific use-case and maybe not the
# the most general implementation. We do need this here for the rest to work, so, if we want,
# this has to be moved to Nabla before we can remove it.
@explicit_intercepts DataFrame Tuple{AbstractVector, AbstractVector} [true, false]
function Nabla.∇(
    ::typeof(DataFrame),
    ::Type{Nabla.Arg{1}},
    _,
    y::DataFrame,
    ȳ::DataFrame,
    x,
    _,
)
    return break_cols(Matrix(ȳ))
end

# Just defining as a function so that we can make it play nice with Nabla
@unionise function break_cols(x::AbstractMatrix)
    return [x[:, i] for i in 1:size(x, 2)]
end

@explicit_intercepts break_cols Tuple{AbstractMatrix} [true]
function Nabla.∇(
    ::typeof(break_cols),
    ::Type{Nabla.Arg{1}},
    _,
    y::Vector,
    ȳ::Vector,
    x,
)
    return hcat(ȳ...)
end

# This is necessary because we can't have `eachindex` called for DataFrames anymore.
function Nabla.zerod_container(x::DataFrame)
    y = Base.copy(x)
    for n in names(y)
        y[:, n] .= Nabla.zerod_container(y[:, n])
    end
    return y
end
function Nabla.oned_container(x::DataFrame)
    y = Base.copy(x)
    for n in names(y)
        y[:, n] .= Nabla.oned_container(y[:, n])
    end
    return y
end
function Nabla.randned_container(x::DataFrame)
    y = Base.copy(x)
    for n in names(y)
        y[:, n] .= Nabla.randned_container(y[:, n])
    end
    return y
end

@unionise function unpack(df::DataFrame, y::Vector)
    v = reshape(y, size(df))
    col_names = names(df)
    cols = break_cols(v)
    return DataFrame(cols, col_names)
end

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
Base.show(io::IO, ::MIME"text/plain", p::Parameter) = show(io, parameter(p))
function Base.show(io::IO, p::Parameter)
    return get(io, :compact, false) ? show(io, parameter(p)) : Base.show_default(io, p)
end

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
Base.isapprox(x, y::Parameter) = false
Base.isapprox(x::Parameter, y) = false

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
    Bounded(p::T, lb, ub, ε::Float64=$_EPSILON_) -> Bounded{T}

Type for parameters that should be kept within lower-bound `lb` and upper-bound `ub` during
the optimisation process.
Optional `ε` is used to map `[lb, ub]` to `[lb + ε, ub - ε]` for numerical stability.
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
function bounded2R(x, lb, ub, ε)
    # Map x from [lb, ub] to [lb + ε, ub - ε].
    x = (x .- lb) ./ (ub .- lb) .* (ub .- lb .- 2 .* ε) .+ lb .+ ε
    # Clip in case x isn't in [lb + ε, ub - ε].
    x = clamp.(x, lb .+ ε, ub .- ε)
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
