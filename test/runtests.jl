using DataFrames
using Dates
using Distributions
using FillArrays
using GPForecasting
using LineSearches
using LinearAlgebra
using Memento
using Nabla
using Optim
using Random: MersenneTwister, seed!
using Test

_ATOL_ = 1e-5


# Keep Logging to a minimum
logger = Memento.config!("warn"; fmt="[{level} | {name}]: {msg}")

@testset "GPForecasting" begin
    include("core/optimise.jl")
    include("pairwise.jl")
    include("core/algebra.jl")
    include("core/node.jl")
    include("core/parameter.jl")
    include("kernel.jl")
    include("mean.jl")
    include("gp.jl")
    include("gaussian.jl")
    include("pdf.jl")
    include("core/nabla.jl")
    include("synthetic.jl")
    include("NN.jl")
end
