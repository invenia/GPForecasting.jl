export Gaussian

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
    U::Wrapped{<:Union{AbstractMatrix, Cholesky}}
end

function Gaussian(
    μ::Wrapped{T},
    Σ::Wrapped{G},
) where {T <: AbstractArray, G <: AbstractArray}
    return Gaussian{T, G}(μ, Σ, Matrix(undef, 0, 0))
end

function Gaussian(
    μ::Adjoint{H, T},
    Σ::Wrapped{G},
) where {H, T <: AbstractArray, G <: AbstractArray}
    return Gaussian{T, G}(T(μ), Σ, Matrix(undef, 0, 0))
end

function Gaussian(
    μ::Adjoint{H, T},
    Σ::Wrapped{G},
    U::Wrapped{<:AbstractMatrix},
) where {H, T <: AbstractArray, G <: AbstractArray}
    return Gaussian{T, G}(T(μ), Σ, U)
end

function Gaussian(
    μ::Wrapped{T},
    Σ::Adjoint{H, G},
) where {H, T <: AbstractArray, G <: AbstractArray}
    return Gaussian{T, G}(μ, G(Σ), Matrix(undef, 0, 0))
end

function Gaussian(
    μ::Wrapped{T},
    Σ::Adjoint{H, G},
    U::Wrapped{<:AbstractMatrix},
) where {H, T <: AbstractArray, G <: AbstractArray}
    return Gaussian{T, G}(μ, G(Σ), U)
end

function Gaussian(
    μ::Adjoint{H, T},
    Σ::Adjoint{H, G},
) where {H, T <: AbstractArray, G <: AbstractArray}
    return Gaussian{T, G}(T(μ), G(Σ), Matrix(undef, 0, 0))
end

function Gaussian(
    μ::Adjoint{H, T},
    Σ::Adjoint{H, G},
    U::Wrapped{<:AbstractMatrix},
) where {H, T <: AbstractArray, G <: AbstractArray}
    return Gaussian{T, G}(T(μ), G(Σ), U)
end

Statistics.mean(g::Gaussian) = g.μ
Statistics.cov(g::Gaussian) = g.Σ
Statistics.var(g::Gaussian) = reshape(diag(cov(g)), size(mean(g), 2), size(mean(g), 1))'

"""
    dim(dist::Gaussian) -> Int

Get the dimensionality of a distribution `dist`.

# Arguments
- `dist::Gaussian`: The distribution of type `Gaussian`.

# Returns
- `Int`: The dimension of the distribution
"""
@unionise Distributions.dim(dist::Gaussian) = length(dist.μ)

Base.size(dist::Gaussian) = size(mean(dist))

"""
    cholesky(dist::Gaussian) -> Cholesky

Compute the Cholesky of the covariance matrix of a MVN `dist`

# Arguments
- `dist::Gaussian`: MVN that contains the covariance matrix to compute the Cholesky of.

# Returns
- `Cholesky`: Computed Cholesky decomposition.
"""
@unionise function LinearAlgebra.cholesky(dist::Gaussian)
    if dist.U == Matrix(undef, 0, 0)
        # NOTE: Adding a tiny regularizer to the main diagonal shifts the eigenvalues
        # and ensures that the matrix is positive definite, which avoids the possibility
        # of a PosDefException
        dist.U = cholesky(Symmetric(dist.Σ) .+ _EPSILON_ .* Eye(dim(dist))).U
    end
    return Cholesky(dist.U, 'U', 0)
end

@unionise function LinearAlgebra.cholesky(dist::Gaussian{T, G}) where {T <: AbstractArray, G <: BlockDiagonal}
    if dist.U == Matrix(undef, 0, 0)
        # extended for BlockDiagonal based on cholesky(::BlockDiagonal)
        dist.U = BlockDiagonal(map(blocks(dist.Σ)) do block
            cholesky(block + _EPSILON_ * Eye(block)).U
        end)
    end
    return Cholesky(dist.U, 'U', 0)
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
function StatsBase.sample(dist::Gaussian, n::Integer=1)
    L = cholesky(dist).L
    if n > 1
        return mean(dist) .+ reshape(L * randn(dim(dist), n), size(dist)..., n)
    else
        return mean(dist) .+ reshape(L * randn(dim(dist), n), size(dist)...)
    end
end
Statistics.rand(dist::Gaussian) = sample(dist)
Statistics.rand(dist::Gaussian, n::Int) = sample(dist, n)

Distributions.MvNormal(d::Gaussian{T}) where {T} = MvNormal(collect(vec(d.μ[:, :]')), d.Σ)

# handles old version of Eye on old versions of FillArrays (with Julia 0.6)
Distributions.MvNormal(d::Gaussian{T, <:Eye}) where {T} = MvNormal(collect(vec(d.μ[:, :]')), collect(d.Σ))

function ModelAnalysis.mll_joint(d::Gaussian{T, G}, y::AbstractMatrix{<:Real}) where {T, G<:BlockDiagonal}
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
