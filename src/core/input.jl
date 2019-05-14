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
    if !isa(x.val, DataFrame)
        return typeof(x)(reshape(x.val[i, :], 1, size(x.val, 2)))
    else
        return typeof(x)(DataFrame(x.val[i, :]))
    end
end

function Base.getindex(x::Input, i::Int)
    return typeof(x)(x.val[i])
end
