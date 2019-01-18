module GPForecasting

import Base: *, +, -, reduce, map, zip, show, getindex, get, isapprox, convert, chol, zero,
size, var, diag, hcat
import Distributions: MvNormal, sample, logpdf
export sample
using Nabla
using LineSearches
using Optim
using Memento
using DataFrames
using Distributions
using Missings
const LOGGER = getlogger(current_module())   # or `get_logger(@__MODULE__)` on 0.7
const _EPSILON_ = 1e-6 # Precision constant
const packagehomedir = dirname(@__DIR__) #dirname ascends the directory...
const Wrapped{T} = Union{T, Node{T}}

function __init__()
    Memento.register(LOGGER) # Register the Logger
end
abstract type Random end

include("core/node.jl")

"""
    Kernel <: AbstractNode

Abstract supertype for all Kernels.
"""
abstract type Kernel <: AbstractNode end

"""
    Mean <: AbstractNode

Abstract supertype for all Means.
"""
abstract type Mean <: AbstractNode end

"""
    Process <: Random

Abstract supertype for all stochastic processes.
"""
abstract type Process <: Random end

"""
    GP{K<:Kernel, M<:Mean} <: Process

Gaussian process.

# Fields:
- `m::Mean`: Mean
- `k::Kernel`: Kernel

# Constructors:
    GP(m::Mean, k::Kernel)

    GP(n::Real, k::Kernel)

Return GP with constant mean `n`.

    GP(k::Kernel)

Return GP with zero mean.
"""
mutable struct GP{K<:Kernel, M<:Mean} <: Process
    m::M
    k::K
end

include("core/optimisedalgebra.jl")

using GPForecasting.OptimisedAlgebra
export BlockDiagonal, blocks

include("core/util.jl")
include("core/input.jl")
include("core/parameter.jl")
include("gaussian.jl")
include("core/pairwise.jl")
include("kernel.jl")
include("multikernel.jl")
include("mean.jl")
include("multimean.jl")
include("gp.jl")
include("core/datahandling.jl")
include("pdf.jl")
include("core/optimise.jl")
include("core/parsing.jl")
include("core/experiment.jl")

include("experiments/Experiments.jl")
using GPForecasting.Experiments

end
