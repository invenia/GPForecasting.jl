"""
    materialize(X)

For values, materialize `Adjoint`- and `Transpose`-wrapped matrices but leave all other
values as-is.
For types, determine the type of a materialized value of the given type.

This is used internally to ensure that `Gaussian`s do not end up holding `Adjoint`- or
`Transpose`-wrapped matrices.
"""
materialize(X::Wrapped{<:Union{Adjoint,Transpose}}) = copy(X)
materialize(X::Wrapped{<:AbstractArray}) = X
materialize(::Type{Wrapped{T}}) where {T} = materialize(T)
materialize(::Type{<:Union{Adjoint{T,S},Transpose{T,S}}}) where {T,S} = materialize(S)
materialize(::Type{T}) where {T<:AbstractArray} = T

"""
    Gaussian

A Gaussian distribution.

# Fields
- `μ`: Mean.
- `Σ`: Covariance.
- `chol`: Cholesky factorization of `Σ`.
"""
mutable struct Gaussian{
    T <: AbstractArray,
    G <: AbstractArray,
} <: Distribution{Matrixvariate, Continuous}
    μ::Wrapped{T}
    Σ::Wrapped{G}
    chol::Union{Wrapped{<:Cholesky}, Nothing}

    function Gaussian(
        μ::Wrapped{T},
        Σ::Wrapped{G},
        chol::Union{Wrapped{<:Cholesky}, Nothing}=nothing,
    ) where {T <: AbstractArray, G <: AbstractArray}
        return new{materialize(T), materialize(G)}(materialize(μ), materialize(Σ), chol)
    end
end

function Gaussian(
    μ::Wrapped{T},
    Σ::Wrapped{G},
    U::Wrapped{<:AbstractMatrix},
) where {T <: AbstractArray, G <: AbstractArray}
    Base.depwarn(
        "`Gaussian(μ, Σ, U)` is deprecated, use `Gaussian(μ, Σ, cholesky(Σ))` instead",
        :Gaussian,
    )
    return Gaussian(μ, Σ, Cholesky(U, 'U', 0))
end

function Base.getproperty(g::Gaussian, x::Symbol)
    if x === :U
        Base.depwarn("`(g::Gaussian).U` is deprecated, use g.chol.U instead", :getproperty)
        return g.chol.U
    else
        return getfield(g, x)
    end
end

# We can't use the default printing for `Distribution`s because it calls `print` on the
# internal fields, which errors for `Gaussian` as it might contain a `nothing`
function Base.show(io::IO, g::Gaussian{T, G}) where {T, G}
    println(io, "Gaussian{", T, ", ", G, "}(")
    print(io, "    μ: ")
    # Use compact and limited printing to ensure we don't spit out entire huge matrices
    show(IOContext(io, :compact=>true, :limit=>true), g.μ)
    print(io, "\n    Σ: ")
    show(IOContext(io, :compact=>true, :limit=>true), g.Σ)
    print(io, "\n    chol: ")
    if g.chol === nothing
        # `nothing` can't be `print`ed, but even if we `show` it, just saying that it's
        # nothing is not particularly informative, so we can instead show what it means
        # for it to be nothing
        print(io, "<not yet computed>")
    else
        show(IOContext(io, :compact=>true, :limit=>true), g.chol)
    end
    print(io, "\n)")
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
Base.size(dist::Gaussian, i::Int) = size(mean(dist), i)

"""
    cholesky(dist::Gaussian) -> Cholesky

Compute the Cholesky of the covariance matrix of a MVN `dist`

# Arguments
- `dist::Gaussian`: MVN that contains the covariance matrix to compute the Cholesky of.

# Returns
- `Cholesky`: Computed Cholesky decomposition.
"""
@unionise function LinearAlgebra.cholesky(dist::Gaussian)
    if dist.chol === nothing
        # NOTE: Adding a tiny regularizer to the main diagonal shifts the eigenvalues
        # and ensures that the matrix is positive definite, which avoids the possibility
        # of a PosDefException
        dist.chol = cholesky(Symmetric(dist.Σ) .+ _EPSILON_ .* Eye(dim(dist)))
    end
    return dist.chol
end

@unionise function LinearAlgebra.cholesky(dist::Gaussian{<:AbstractArray, <:BlockDiagonal})
    if dist.chol === nothing
        # extended for BlockDiagonal based on cholesky(::BlockDiagonal)
        U = BlockDiagonal(map(blocks(dist.Σ)) do block
            cholesky(block + _EPSILON_ * Eye(block)).U
        end)
        dist.chol = Cholesky(U, 'U', 0)
    end
    return dist.chol
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
        return -logpdf(d, y) / length(y)
    elseif d.chol !== nothing && isa(d.chol.U, BlockDiagonal)
        return sum([-logpdf(Gaussian(
            reshape(d.μ[i, :], 1, size(d.μ, 2)),
            blocks(d.Σ)[i],
            Cholesky(blocks(d.chol.U)[i], 'U', 0),
        ), reshape(y[i, :], 1, size(y, 2))) for i in 1:length(blocks(d.Σ))]) / length(y)
    else
        return sum([-logpdf(Gaussian(
            reshape(d.μ[i, :], 1, size(d.μ, 2)),
            blocks(d.Σ)[i]
        ), reshape(y[i, :], 1, size(y, 2))) for i in 1:length(blocks(d.Σ))]) / length(y)
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
