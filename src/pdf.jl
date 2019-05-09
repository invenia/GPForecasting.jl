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
@unionise function Distributions.logpdf(dist::Gaussian, x::AbstractArray)
    L = cholesky(dist).L
    log_det = 2sum(log, diag(L))
    if size(x, 2) > 1 && size(L, 2) == length(x) # This means that the covariance matrix has entries for
    # all outputs and timestamps.
        z = L \ vec((x .- dist.μ)')
    elseif size(L, 2) == size(x, 2) # This means we have a covariance matrix that has entries
    # only for the different outputs, but for a single timestamp. This allows for the
    # automatic computation of the logpdf of a set of realisations, i.e. p(x[1, :], ... x[n, :]|dist)
        z = L \ (x .- dist.μ')'
        return (-size(x, 1) * (log_det + size(x, 2) * log(2π)) - sum(abs2, z)) / 2
    else
        z = L \ (x .- dist.μ)
    end
    return -(log_det + length(x) * log(2π) + sum(abs2, z)) / 2
end

# This looks quite redundant, but is necessary to remove the ambiguity introduced above due
# to the unionise, since Distributions.jl has its own logpdf methods that can be as
# especialised as the above.
function Distributions.logpdf(dist::Gaussian, x::AbstractMatrix{<:Real})
    L = cholesky(dist).L
    log_det = 2sum(log, diag(L))
    if size(x, 2) > 1 && size(L, 2) == length(x) # This means that the covariance matrix has entries for
    # all outputs and timestamps.
        z = L \ vec((x .- dist.μ)')
    elseif size(L, 2) == size(x, 2) # This means we have a covariance matrix that has entries
    # only for the different outputs, but for a single timestamp. This allows for the
    # automatic computation of the logpdf of a set of realisations, i.e. p(x[1, :], ... x[n, :]|dist)
        z = L \ (x .- dist.μ')'
        return (-size(x, 1) * (log_det + size(x, 2) * log(2π)) - sum(abs2, z)) / 2
    else
        z = L \ (x .- dist.μ)
    end
    return -(log_det + length(x) * log(2π) + sum(abs2, z)) / 2
end

@unionise function Distributions.logpdf(
    gp::GP,
    x,
    y::AbstractArray{<:Real},
    params::Vector{G}
) where {G <: Real}
    ngp = GP(gp.m, set(gp.k, params)) # update kernels with new parameters
    # if we want to update the means as well, we should overload this.
    return logpdf(ngp::GP, x, y)
end

@unionise function Distributions.logpdf(
   gp::GP{K, M},
    x,
    y::AbstractMatrix{<:Real},
    params::Vector{G}
) where {K <: LMMKernel, M <: Mean, G <: Real}
    ngp = GP(gp.m, set(gp.k, params)) # update kernels with new parameters
    # if we want to update the means as well, we should overload this.
    return logpdf(ngp::GP, x, y)
end

@unionise function Distributions.logpdf(
    gp::GP{K, U},
    x,
    y::AbstractMatrix{<:Real}
) where {K <: LMMKernel, U <: Mean} # Assuming always zero mean here. We should properly dispatch later

    yt = y'
    n_d = size(x, 1)
    m = unwrap(gp.k.m)
    p = unwrap(gp.k.p)
    σ² = unwrap(gp.k.σ²)
    σ² = isa(σ², Union{Float64, Nabla.Branch{Float64}}) ? Ones(p, 1) * σ² : reshape(σ², p, 1)
    H = float(unwrap(gp.k.H)) # Prevents Nabla from breaking in case H has Ints.

    Kd = (kern(x) for kern in gp.k.ks)

    yiσ² = yt ./ σ²
    HiΛy = reshape(transpose(H) * yiσ², n_d * m, 1)
    Ls = (cholesky(Symmetric(K) .+ _EPSILON_ .* Eye(n_d)).U for K in Kd)
    LQ = sum_kron_J_ut(m, Ls...)
    M = cholesky(
        Symmetric(eye_sum_kron_M_ut(transpose(H) * (H ./ σ²), Ls...)) .+
        _EPSILON_ .* Eye(m * n_d)
    ).L
    log_det = n_d * sum(log, σ²) + 2sum(log, diag(M))
    z = M \ (LQ * HiΛy)
    return -(n_d * p * log(2π) + log_det + dot(yiσ², yt) - sum(abs2, z)) / 2
end

@unionise function Distributions.logpdf(dist::Gaussian, xs::Vector{<:Vector})
    L = cholesky(dist).L
    log_det = 2sum(log, diag(L))
    lpdf = zero(typeof(log_det))
    for x in xs
        z = L \ (x .- dist.μ)
        lpdf += -(log_det + length(x) * log(2π) + sum(abs2, z)) / 2
    end
    return lpdf
end

@unionise function optlogpdf(
    gp::GP{K, U},
    x,
    y::AbstractMatrix{<:Real}
) where {K <: OLMMKernel, U <: Mean}
    n = size(x, 1)
    p = unwrap(gp.k.p)
    m = unwrap(gp.k.m)
    σ² = Ones(p) .* unwrap(gp.k.σ²)
    H = float(unwrap(gp.k.H)) # Prevents Nabla from breaking in case H has Ints.
    d = unwrap(gp.k.D)
    D = Ones(m) .* d
    P = unwrap(gp.k.P)

    Σn = Diagonal(σ²) .+ H * Diagonal(D) * H'
    gn = Gaussian(Zeros(p), Σn)


    # Noise contributions
    # These decouple timestamps, so we can compute one at a time.
    lpdf = logpdf(gn, y)

    # Latent process contributions
    # These decouple amongst different latent processes, so we can compute one at time.
    yl = y * P'
    Σlk = gp.k.ks(x)
    proj_noise = (unwrap(gp.k.σ²) + d) * Eye(n)
    glk = Gaussian(Zeros(n), proj_noise + Σlk)
    gln = Gaussian(Zeros(n), proj_noise)
    lpdf += logpdf(glk, yl')
    lpdf -= logpdf(gln, yl')
    return lpdf
end

@unionise function Distributions.logpdf(
    gp::GP{K, U},
    x,
    y::AbstractMatrix{<:Real}
) where {K <: OLMMKernel, U <: Mean}

    n = size(x, 1)
    p = unwrap(gp.k.p)
    m = unwrap(gp.k.m)
    σ² = Ones(p) .* unwrap(gp.k.σ²)
    H = float(unwrap(gp.k.H)) # Prevents Nabla from breaking in case H has Ints.
    D = unwrap(gp.k.D)
    S_sqrt = unwrap(gp.k.S_sqrt)
    isa(gp.k.ks, Kernel) && !isa(D, Vector) && S_sqrt ≈ Ones(m) && return optlogpdf(gp, x, y)

    D = isa(D, Vector) ? D : Ones(m) .* D
    P = unwrap(gp.k.P)

    Σn = Diagonal(σ²) .+ H * Diagonal(D) * H'
    gn = Gaussian(Zeros(p), Σn)

    # Noise contributions
    # These decouple timestamps, so we can compute one at a time.
    lpdf = logpdf(gn, y)

    # Latent process contributions
    # These decouple amongst different latent processes, so we can compute one at time.
    yl = y * P'
    for i in 1:m
        proj_noise = (unwrap(gp.k.σ²)/(S_sqrt[i])^2 + D[i]) * Eye(n)
        Σlk = gp.k.ks[i](x)
        glk = Gaussian(Zeros(n), proj_noise + Σlk)
        gln = Gaussian(Zeros(n), proj_noise)
        yls = yl[:, i]
        lpdf += logpdf(glk, yls) - logpdf(gln, yls)
    end
    return lpdf
end

@unionise function Distributions.logpdf(gp::GP, x, y::AbstractArray{<:Real})
    return logpdf(gp(x), y)
end

"""
    objective(gp::GP, x, y::AbstractArray) -> Function

Objective function that, when minimised, yields maximum probability of observations `y` for
a `gp` evaluated at points `x`. Returns a function of the `GP` parameters.
"""
@unionise function objective(gp::GP, x, y::AbstractArray{<:Real})
    return function f(params)
        return -logpdf(gp::GP, x, y, params)
    end
end

"""
    titsiasELBO(gp::GP, x, y::AbstractArray{<:Real})

Compute the lower bound for the posterior logpdf under Titsias' approach. See:
"Variational Learning of Inducing Variables in Sparse Gaussian Processes"
"""
@unionise function titsiasELBO(gp::GP, x, y::AbstractArray{<:Real})
    Xm = unwrap(gp.k.Xm)
    k = gp.k.k
    m = gp.m
    σ² = unwrap(gp.k.σ²)
    num_m = unwrap(gp.k.n)
    # Compute Qnn
    Kmm = k(Xm, Xm)
    Kmn = k(Xm, x)
    Umm = cholesky(Kmm + _EPSILON_^2 * Eye(num_m)).U
    Q_sqrt = Umm' \ Kmn
    Qnn = Q_sqrt' * Q_sqrt
    # Compute first term
    log_N = logpdf(Gaussian(m(x), Qnn + σ² * Eye(size(Qnn, 1))), y)
    # Compute K̅
    return log_N - (2 * σ²)^(-1) * sum(var(k, x)) - tr(Qnn)
end

@unionise function titsiasELBO(
    gp::GP{SparseKernel{OLMMKernel}, <:Mean},
    x,
    y::AbstractMatrix{<:Real}
)
    n = size(x, 1)
    p = unwrap(gp.k.k.p)
    m = unwrap(gp.k.k.m)
    σ² = ones(p) .* unwrap(gp.k.k.σ²)
    H = float.(unwrap(gp.k.k.H)) # Prevents Nabla from breaking in case H has Ints.
    D = unwrap(gp.k.k.D)
    S_sqrt = unwrap(gp.k.k.S_sqrt)

    # For now we won't have a single kernel optimisation for this case. Shouldn't be an
    # issue, as we saw single kernel models don't give good results even for the full model
    # isa(gp.k.k.ks, Kernel) && !isa(D, Vector) && S_sqrt ≈ ones(m) && return optlogpdf(gp, x, y)

    D = isa(D, Vector) ? D : ones(m) .* D
    P = unwrap(gp.k.k.P)

    Σn = Diagonal(σ²) .+ H * Diagonal(D) * H'
    gn = Gaussian(zeros(p), Σn)
    lpdf = 0.0

    # Noise contributions
    # These decouple timestamps, so we can compute one at a time.
    lpdf += logpdf(gn, y)

    # Latent process contributions
    # These decouple amongst different latent processes, so we can compute one at time.
    yl = y * P'
    # By having the Xm defined like this, we make it such that the same inducing points are
    # used for all latent processes.
    Xm = unwrap(gp.k.Xm)
    sσ² = unwrap(gp.k.σ²)
    num_m = unwrap(gp.k.n)
    for i in 1:m
        proj_noise = (unwrap(gp.k.σ²)/(S_sqrt[i])^2 + D[i]) * Eye(n)
        # Here we compute the sparse GP contributions
        Kmm = gp.k.k.ks[i](Xm, Xm)
        Kmn = gp.k.k.ks[i](Xm, x)
        Umm = cholesky(Kmm + _EPSILON_^2 * Eye(num_m)).U
        Q_sqrt = Umm' \ Kmn
        Qnn = Q_sqrt' * Q_sqrt
        log_N = logpdf(Gaussian(Zeros(n), Qnn + sσ² * Eye(n) + proj_noise), yl[:, i])
        gln = Gaussian(zeros(n), proj_noise)
        # The implementation below is better, but leads to Nabla issues. TODO: make it work,
        slpdf = log_N - (2 * sσ²)^(-1) * (sum(var(gp.k.k.ks[i], x)) - tr(Qnn))
        lpdf += slpdf - logpdf(gln, yl[:, i])
    end
    return lpdf
end

@unionise function titsiasELBO(gp::GP, x, y::AbstractArray{<:Real}, params::Vector{<:Real})
    ngp = GP(gp.m, set(gp.k, params)) # update kernels with new parameters
    # if we want to update the means as well, we should overload this.
    return titsiasELBO(ngp::GP, x, y)
end

"""
    titsiasobj(gp::GP, x, y::AbstractArray{<:Real}, Xm, σ²)

Return objective function for learning a sparse GP under Titsias' approximations. Passing
this to `learn_sparse` automatically adopts the approach as specified in "Variational
Learning of Inducing Variables in Sparse Gaussian Processes". Note that this should receive
a regular GP, which will be made sparse.
"""
@unionise function titsiasobj(gp::GP, x, y::AbstractArray{<:Real}, Xm, σ²)
    sk = SparseKernel(gp.k, Xm, σ²)
    return function f(params)
        return -titsiasELBO(GP(gp.m, sk), x, y, params)
    end
end
# TODO: A method that let's specify only the number of inducing points
