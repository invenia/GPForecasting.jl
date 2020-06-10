### Functions required for differentiable EF
import Base.rem, Statistics.cov

# Nabla explicit_intercepts for reminder function
@explicit_intercepts rem Tuple{Float64, Float64} [true, true]
Nabla.∇(::typeof(rem), ::Type{Arg{1}}, _, z, z̄, x::Float64, y::Float64) = 0.0
Nabla.∇(::typeof(rem), ::Type{Arg{2}}, _, z, z̄, x::Float64, y::Float64) = 0.0

# Nabla explicit_intercepts for conditioned covariance
@explicit_intercepts cov Tuple{CovarianceEstimator, AbstractArray} [false, true]
Nabla.∇(::typeof(cov), ::Type{Arg{2}}, _, z, z̄, x::CovarianceEstimator, y::AbstractArray) = zeros(size(y))

# Kronecker delta kernel and corresponding functions
mutable struct Kronecker <: Kernel
    τ
    Kronecker(τ) = isconstrained(τ) ? new(τ) : new(Positive(τ))
end
function (k::Kronecker)(x::GPForecasting.ArrayOrReal, y::GPForecasting.ArrayOrReal)
    return map(z->z==0 ? 1.0 : 0.0, (x .- y') .% GPForecasting.unwrap(k.τ))
end
(k::Kronecker)(x::GPForecasting.ArrayOrReal) = k(x, x)

# Weight kernel and corresponding functions
mutable struct Weight <: Kernel
    λ
    τ
    function Weight(λ, τ)
        l = isconstrained(λ) ? λ : Positive(λ)
        t = isconstrained(τ) ? τ : Positive(τ)
        return new(l, t)
    end
end
function (k::Weight)(x::GPForecasting.ArrayOrReal, y::GPForecasting.ArrayOrReal)
    return GPForecasting.unwrap(k.λ) .^ (GPForecasting.pairwise_dist(x, y) ./ GPForecasting.unwrap(k.τ))
end
(k::Weight)(x::GPForecasting.ArrayOrReal) = k(x, x)

# EF object and corresponding function
mutable struct EF
    k::Kernel
    estimator::CovarianceEstimator
    x
    y
end

@unionise function (ef::EF)(x_test)
    w = ef.k(ef.x, x_test)[:]
    w /= sum(w)
    μ = w' * ef.y
    σ = sqrt.(( w' * (ef.y .- μ).^2 ) ./ (1.0 - dot(w, w)))
    C = cov(ef.estimator, ef.y ./ std(ef.y; dims=1))
    Σ = σ' .* C .* σ
    return (μ[:], Symmetric(Σ))
end
