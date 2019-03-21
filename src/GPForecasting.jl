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
    RootLog,
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

# core/pairwise
export pairwise_dist, sq_pairwise_dist

# core/input
export Input, Observed, Latent

# core/util
export cov_EB, cov_LW

# core/optimise
export minimise, learn, learn_summary, minimise_summary

# core/parameter
export Bounded, DynamicBound, Fixed, Named, Parameter, Positive, isconstrained

using DataFrames
using Distributions
using FillArrays
using LineSearches
using LinearAlgebra
using Memento
using ModelAnalysis
using Nabla
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
using .OptimisedAlgebra


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

end  # module
