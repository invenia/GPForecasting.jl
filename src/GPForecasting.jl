module GPForecasting

export sample

# gp.jl
export Process, GP, condition, condition_sparse, credible_interval

# gaussian.jl
export Gaussian, marginal_mean_logloss, joint_mean_logloss

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
export balanced_return_obj,
    normalised_expected_posterior_return,
    normalised_expected_posterior_return_balanced,
    normalised_expected_posterior_return_balanced_obj,
    normalised_expected_posterior_return_obj,
    normalised_expected_return,
    normalised_expected_return_balanced,
    logpdf,
    map_obj,
    mle_obj,
    reglogpdf,
    return_obj,
    titsiasobj,
    mll_pointwise_posterior_obj

# kernel.jl
export ▷,
    ←,
    ↻,
    BinaryKernel,
    CosineKernel,
    ConstantKernel,
    DiagonalKernel,
    DotKernel,
    elwise,
    EQ,
    HazardKernel,
    HeteroskedasticDiagonalKernel,
    Kernel,
    LinearLayer,
    MA,
    ManifoldKernel,
    NKN,
    PeriodicKernel,
    PosteriorKernel,
    ProductKernel,
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
export GOLMMKernel,
    LMMKernel,
    LMMPosKernel,
    LSOLMMKernel,
    MultiKernel,
    MultiOutputKernel,
    NaiveLMMKernel,
    NoiseKernel,
    OLMMKernel
    # verynaiveLMMKernel

# core/pairwise
export elwise_dist, pairwise_dist, sq_elwise_dist, sq_pairwise_dist

# core/input
export Input, Observed, Latent

# core/optimise
export minimise, learn, learn_summary, learn_sparse, minimise_summary

# core/parameter
export Bounded, DynamicBound, Fixed, Named, Parameter, Positive, isconstrained

#NN.jl
export GPFNN,
    NNLayer,
    BatchNormLayer,
    ProductLayer,
    relu,
    noisy_relu,
    leaky_relu,
    softplus,
    sigmoid

# reexport so as not be breaking from when BlockDiagonals was part of this package.
export BlockDiagonal, blocks

using BlockDiagonals: BlockDiagonal, blocks
using DataFrames
using Distances
using Distributions
using FillArrays: Eye, Ones, Zeros
using LineSearches
using LinearAlgebra
using Memento
using Metrics
using Nabla
using Optim
using ParamTracer
using ParamTracer: @trace, register
using Random
using SparseArrays
using Statistics
using StatsBase

const MODULE = "GPForecasting"

function __init__()
    Memento.register(LOGGER) # Register the Logger
    ParamTracer.register(["$MODULE.Optim"])
end

const LOGGER = getlogger(@__MODULE__)
const _EPSILON_ = 1e-6 # Precision constant
const packagehomedir = dirname(@__DIR__) #dirname ascends the directory...
const Wrapped{T} = Union{T, Node{T}}

@deprecate(∿, ↻, true)

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

include("core/input.jl")
include("core/parameter.jl")
include("gaussian.jl")
include("core/pairwise.jl")
include("NN.jl")
include("kernel.jl")
include("multikernel.jl")
include("mean.jl")
include("multimean.jl")
include("gp.jl")
include("pdf.jl")
include("core/optimise.jl")

end  # module
