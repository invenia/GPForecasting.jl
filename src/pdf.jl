export logpdf, objective

"""
    logpdf(dist::Gaussian, x::AbstractArray) -> Float64

Compute the log–pdf of a Gaussian distribution `dist` for a design matrix `x`.

# Arguments
- `dist::Gaussian`: The input distribution.
- `x::{AbstractArray, Float64}`: The x at which to evaluate the log pdf.

# Returns
- `Float64`: The logpdf value of the distribution evaluated at `x`.

    logpdf(gp::GP, x::AbstractArray, y::AbstractArray) -> Float64

Build a finite dimensional distribution of `gp` over points `x` and then compute the log-pdf
for a design matrix `y`.

# Arguments
- `gp::GP`: The input process.
- `x::Vector{Float64}`: The `x` at which to evaluate the `GP`.
- `y::AbstractArray`: The observation values to be used in computing the log-pdf.

# Returns
- `Float64`: The logpdf value of the distribution evaluated at `y`.

    logpdf(
        gp::GP,
        x,
        y::AbstractArray,
        params::Vector{G}
    ) where {G <: Real} -> Float64

Update `gp` parameter values with `params` and then call logpdf(ngp, x, y), where `ngp` is
the updated `GP`. Does NOT affect `gp`.
"""
@unionise function logpdf(dist::Gaussian, x::AbstractArray)
    U = chol(dist)
    if size(x, 2) > 1
        z = U' \ (x .- dist.μ)'[:]
    else
        z = U' \ (x .- dist.μ)
    end
    log_det = 2 * sum(log.(diag(U)))
    return -.5 * (log_det + prod(size(x)) * log(2π) + dot(z, z))
end

# This looks quite redundant, but is necessary to remove the ambiguity introduced above due
# to the unionise, since Distributions.jl has its own logpdf methods that can be as
# especialised as the above.
function logpdf(dist::Gaussian, x::AbstractMatrix{<:Real})
    U = chol(dist)
    z = U' \ (x .- dist.μ)'[:]
    log_det = 2 * sum(log.(diag(U)))
    return -.5 * (log_det + prod(size(x)) * log(2π) + dot(z, z))
end

@unionise function logpdf(
    gp::GP,
    x,
    y::AbstractArray,
    params::Vector{G}
) where {G <: Real}
    ngp = GP(gp.m, set(gp.k, params)) # update kernels with new parameters
    # if we want to update the means as well, we should overload this.
    return logpdf(ngp::GP, x, y)
end

@unionise function logpdf(
   gp::GP{K, M},
    x,
    y::AbstractArray,
    params::Vector{G}
) where {K <: LMMKernel, M <: Mean, G <: Real}
    ngp = GP(gp.m, set(gp.k, params)) # update kernels with new parameters
    # if we want to update the means as well, we should overload this.
    return logpdf(ngp::GP, x, y)
end

@unionise function logpdf(
    gp::GP{K, U},
    x,
    y::AbstractArray
) where {K <: LMMKernel, U <: Mean} # Assuming always zero mean here. We should properly dispatch later

    yt = y'
    n_d = size(x, 1)
    m = unwrap(gp.k.m)
    p = unwrap(gp.k.p)
    σ² = unwrap(gp.k.σ²)
    σ² = isa(σ², Union{Float64, Nabla.Branch{Float64}}) ? ones(p, 1) * σ² : reshape(σ², p, 1)
    H = float.(unwrap(gp.k.H)) # Prevents Nabla from breaking in case H has Ints.

    Kd = [kern(x) for kern in gp.k.ks]

    yiσ² = yt ./ σ²
    HiΛy = reshape(At_mul_B(H, yiσ²), n_d * m, 1)
    Ls = [chol(Symmetric(K) .+ _EPSILON_ .* eye(n_d)) for K in Kd]
    LQ = sum_kron_J_ut(m, Ls...)
    M = chol(
        Symmetric(eye_sum_kron_M_ut(At_mul_B(H, H ./ σ²), Ls...)) .+
        _EPSILON_ .* eye(m * n_d)
    )
    log_det = n_d * sum(log.(σ²)) + 2sum(log.(diag(M)))
    z = M' \ (LQ * HiΛy)
    return -.5(n_d * p * log(2π) + log_det + dot(yiσ², yt) - dot(z, z))
end

@unionise function logpdf(
    gp::GP{K, U},
    x,
    y::AbstractArray
) where {K <: OLMMKernel, U <: Mean}
    n = size(x, 1)
    p = unwrap(gp.k.p)
    m = unwrap(gp.k.m)
    σ² = ones(p) .* unwrap(gp.k.σ²)
    H = float.(unwrap(gp.k.H)) # Prevents Nabla from breaking in case H has Ints.
    D = unwrap(gp.k.D)
    D = isa(D, Vector) ? D : ones(m) .* D
    P = unwrap(gp.k.P)
    S_sqrt = unwrap(gp.k.S_sqrt)

    Σn = diagm(σ²) .+ H * diagm(D) * H'
    gn = Gaussian(zeros(p), Σn)
    lpdf = 0.0

    # Noise contributions
    # These decouple timestamps, so we can compute one at a time.
    for i in 1:n
        lpdf += logpdf(gn, y[i, :])
    end

    # Latent process contributions
    # These decouple amongst different latent processes, so we can compute one at time.
    yl = y * P'
    for i in 1:m
        proj_noise = (unwrap(gp.k.σ²)/S_sqrt[i] + D[i]) * eye(n)
        Σlk = gp.k.ks[i](x)
        glk = Gaussian(zeros(n), proj_noise + Σlk)
        gln = Gaussian(zeros(n), Σlk)
        lpdf += logpdf(glk, y[:, i]) - logpdf(gln, y[:, i])
    end
    return lpdf
end

@unionise function logpdf(gp::GP, x, y::AbstractArray)
    return logpdf(gp(x), y)
end

"""
    objective(gp::GP, x, y::AbstractArray) -> Function

Objective function that, when minimised, yields maximum probability of observations `y` for
a `gp` evaluated at points `x`. Returns a function of the `GP` parameters.
"""
@unionise function objective(gp::GP, x, y::AbstractArray)
    return function f(params)
        return -logpdf(gp::GP, x, y, params)
    end
end