module GPForecasting

import Base: *, +, -, ^, reduce, map, zip, show, getindex, get, isapprox, convert, zero,
size, hcat
using Compat: Compat, @__MODULE__, tr, undef
using Compat.Distributed: pmap
import Compat.LinearAlgebra: diag
using Compat.LinearAlgebra
using Compat.Random
using Compat.SparseArrays
import Distributions: MvNormal, sample, logpdf

export sample

using DataFrames
using Distributions
using FillArrays
using Missings
using Nullables
using LineSearches
using Memento
using Missings
using Nabla
using Optim

const LOGGER = getlogger(@__MODULE__)
const _EPSILON_ = 1e-6 # Precision constant
const packagehomedir = dirname(@__DIR__) #dirname ascends the directory...
const Wrapped{T} = Union{T, Node{T}}

if VERSION >= v"0.7"
    import LinearAlgebra: LinearAlgebra, adjoint, Adjoint, mul!

    A_mul_Bt(A, B) = A * transpose(B)
    sumdims(A, dims) = sum(A, dims=dims)
    meandims(A, dims) = mean(A, dims=dims)
    stddims(A, dims) = std(A, dims=dims)
    covdims(A, dims; kwargs...) = cov(A; dims=dims, kwargs...)
else
    import Base: A_mul_Bt

    sumdims(A, dims) = sum(A, dims)
    meandims(A, dims) = mean(A, dims)
    stddims(A, dims) = std(A, dims)
    covdims(A, dims; kwargs...) = cov(A, dims; kwargs...)
end

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


end
