using GPForecasting
using Compat.Test

using GPForecasting: sumdims, meandims, stddims

import Compat
using Compat: tr, undef, repeat
using Compat.Dates
using Compat.Distributed
using Compat.LinearAlgebra
using CSV
using DataFrames
using Distributions
using FDM
using FillArrays
using LineSearches
using Memento
using Nabla
using Nullables

_ATOL_ = 1e-5

if isdefined(Compat.Random, :seed!)
    using Compat.Random: seed!
else
    const seed! = Compat.Random.srand
end

if VERSION >= v"0.7"
    proddims(A, dims) = prod(A, dims=dims)
else
    proddims(A, dims) = prod(A, dims)
end


# Write your own tests here.
@testset "GPForecasting.jl" begin

    # Keep Logging to a minimum
    logger = Memento.config!("warn"; fmt="[{level} | {name}]: {msg}")

    include("core/util.jl")
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
end
