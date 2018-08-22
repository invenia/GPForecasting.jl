export Input, Observed, Latent

"""
    Input

Abstract type used for dispatching `NoisyKernel`s.
"""
abstract type Input end

size(x::Input) = size(x.val)
size(x::Input, i::Int) = size(x.val, i)

"""
    Observed <: Input

Type for inputs that correspond to noisy observations.
"""
type Observed <: Input
    val
end

"""
    Latent <: Input

Type for inputs that correspond to non-noisy observations.
"""
type Latent <: Input
    val
end
