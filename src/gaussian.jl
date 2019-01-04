export Gaussian

import Base: chol, mean, cov, var, size
import Distributions: rand, dim
import ModelAnalysis: mll_joint

"""
    Gaussian <: Random

A Gaussian distribution.

# Fields
- `μ`: Mean.
- `Σ`: Covariance.
- `U`: `Σ`'s upper triangular Cholesky decomposition.
"""
mutable struct Gaussian{
    T <: AbstractArray,
    G <: AbstractArray,
} <: Distribution{Matrixvariate, Continuous}
    μ::Wrapped{T}
    Σ::Wrapped{G}
    U::Wrapped{<:AbstractMatrix}
end

function Gaussian(
    μ::Wrapped{T},
    Σ::Wrapped{G},
) where {T <: AbstractArray, G <: AbstractArray}
    return Gaussian{T, G}(μ, Σ, Matrix(0, 0))
end

mean(g::Gaussian) = g.μ
cov(g::Gaussian) = g.Σ
var(g::Gaussian) = reshape(diag(cov(g)), size(mean(g), 2), size(mean(g), 1))'

"""
    dim(dist::Gaussian) -> Int

Get the dimensionality of a distribution `dist`.

# Arguments
- `dist::Gaussian`: The distribution of type `Gaussian`.

# Returns
- `Int`: The dimension of the distribution
"""
@unionise dim(dist::Gaussian) = length(dist.μ)

size(dist::Gaussian) = size(mean(dist))

"""
    chol(dist::Gaussian) -> AbstractMatrix

Compute the Cholesky of the covariance matrix of a MVN `dist`

# Arguments
- `dist::Gaussian`: MVN that contains the covariance matrix to compute the Cholesky of.

# Returns
- `AbstractMatrix`: Computed Cholesky decomposition.
"""
@unionise function chol(dist::Gaussian)
    if dist.U == Matrix(0, 0)
        dist.U = chol(Symmetric(dist.Σ) .+ _EPSILON_ .* eye(dim(dist)))
    end
    return dist.U
end

@unionise function chol(dist::Gaussian{T, G}) where {T <: AbstractArray, G <: BlockDiagonal}
    if dist.U == Matrix(0, 0)
        dist.U = BlockDiagonal(chol.(Symmetric.(blocks(dist.Σ)) .+ _EPSILON_ .* eye.(blocks(dist.Σ))))
    end
    return dist.U
end

"""
    sample(dist::Gaussian, n::Integer=1) -> AbstractMatrix{<:Real}

Sample `n` samples from a MVN `dist`.

# Arguments
- `dist::Gaussian`: Gaussian distribution to sample from.
- `n::Integer`: Number of samples to take.

# Returns
- `AbstractMatrix{<:Real}`: Samples where the columns correspond to different samples.
"""
function sample(dist::Gaussian, n::Integer=1)
    U = chol(dist)
    if n > 1
        return mean(dist) .+ reshape(U' * randn(dim(dist), n), size(dist)..., n)
    else
        return mean(dist) .+ reshape(U' * randn(dim(dist), n), size(dist)...)
    end
end
rand(dist::Gaussian) = sample(dist)
rand(dist::Gaussian, n::Int) = sample(dist, n)

MvNormal(d::Gaussian) = MvNormal(vec(d.μ[:, :]'), d.Σ)

function mll_joint(d::Gaussian{T, G}, y::AbstractMatrix{<:Real}) where {T, G<:BlockDiagonal}
    if length(blocks(d.Σ)) != size(y, 1) # Not sure why one would ever do this, but anyway
        return -logpdf(d, y) / prod(size(y))
    elseif isa(d.U, BlockDiagonal)
        return sum([-logpdf(Gaussian(
            reshape(d.μ[i, :], 1, size(d.μ, 2)),
            blocks(d.Σ)[i],
            blocks(d.U)[i]
        ), reshape(y[i, :], 1, size(y, 2))) for i in 1:length(blocks(d.Σ))]) / prod(size(y))
    else
        return sum([-logpdf(Gaussian(
            reshape(d.μ[i, :], 1, size(d.μ, 2)),
            blocks(d.Σ)[i]
        ), reshape(y[i, :], 1, size(y, 2))) for i in 1:length(blocks(d.Σ))]) / prod(size(y))
    end
end

"""
    hourly_distributions(g::Gaussian)

Break a `Gaussian` into a vector of `MvNormal`s corresponding to each individual timestamp.
NOTE: will NOT work unless the `Gaussian` has a `BlockDiagonal` type covariance.
"""
function hourly_distributions(g::Gaussian)
    throw(
        ArgumentError("The `Gaussian` object must have a `BlockDiagonal` type covariance")
    )
end

function hourly_distributions(g::Gaussian{<:AbstractArray, <:BlockDiagonal})
    size(mean(g), 1) != length(blocks(cov(g))) && throw(DimensionMismatch(
        "Number of timesteps does not correspond to number of covariance blocks"
    ))
    return [MvNormal(mean(g)[i, :], blocks(cov(g))[i]) for i in 1:size(mean(g), 1)]
end
