module GPForecasting

export BlockDiagonal, blocks, sample

# gp.jl
export Process, GP, condition, credible_interval

# gaussian.jl
export Gaussian

# mean.jl
export ConstantMean,
    FunctionMean,
    Mean,
    PosteriorMean,
    ProductMeant,
    ScaledMean,
    SumMean,
    ZeroMean

# multimean.jl
export LMMPosMean, MultiMean, MultiOutputMean, OLMMPosMean


# pdf.jl
export logpdf, objective

# kernel.jl
export ▷,
    ←,
    ∿,
    BinaryKernel,
    ConstantKernel,
    DiagonalKernel,
    DotKernel,
    EQ,
    HazardKernel,
    Kernel,
    MA,
    PeriodicKernel,
    PosteriorKernel,
    RQ,
    ScaledKernel,
    SimilarHourKernel,
    SpecifiedQuantityKernel,
    StretchedKernel,
    SumKernel,
    ZeroKernel,
    hourly_cov,
    isMulti,
    periodicise,
    set,
    stretch

# multikernel.jl
export LMMKernel,
    LMMPosKernel,
    MultiKernel,
    MultiOutputKernel,
    NaiveLMMKernel,
    NoiseKernel,
    OLMMKernel,
    verynaiveLMMKernel


using DataFrames
using Distributions
using FillArrays
using LineSearches
using LinearAlgebra
using Memento
using ModelAnalysis
using Nabla
using Nullables
using Optim
using Random
using SparseArrays
using Statistics
using StatsBase

const LOGGER = getlogger(@__MODULE__)
const _EPSILON_ = 1e-6 # Precision constant
const packagehomedir = dirname(@__DIR__) #dirname ascends the directory...
const Wrapped{T} = Union{T, Node{T}}

__init__() = Memento.register(LOGGER)  # Register the Logger

"""
    Process

Abstract supertype for all stochastic processes.
"""
abstract type Process end

# must be included after defining `Process` and before subtyping `AbstractNode`
include("core/node.jl")
include("core/optimisedalgebra.jl")
using GPForecasting.OptimisedAlgebra

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
include("pdf.jl")
include("core/optimise.jl")


end
