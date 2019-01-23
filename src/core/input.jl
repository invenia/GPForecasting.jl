export Input, Observed, Latent

"""
    Input

Abstract type used for dispatching `NoisyKernel`s.
"""
abstract type Input end

Base.size(x::Input) = size(x.val)
Base.size(x::Input, i::Int) = size(x.val, i)

"""
    Observed <: Input

Type for inputs that correspond to noisy observations.
"""
struct Observed <: Input
    val
end

"""
    Latent <: Input

Type for inputs that correspond to non-noisy observations.
"""
struct Latent <: Input
    val
end

function Base.getindex(x::Input, i::Int, j::Colon)
    return typeof(x)(x.val[i, :])
end

function Base.getindex(x::Input, i::Int)
    return typeof(x)(x.val[i])
end
