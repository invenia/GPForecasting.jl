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

function Distributions.logpdf(
    d::Gaussian{T, G},
    y::AbstractMatrix{<:Real},
) where {T<:Wrapped{<:AbstractArray}, G<:BlockDiagonal}
    if d.chol !== nothing && isa(d.chol.U, BlockDiagonal)
        return sum(
            [
                logpdf(
                    Gaussian(
                        reshape(d.μ[i, :], 1, size(d.μ, 2)),
                        blocks(d.Σ)[i],
                        Cholesky(blocks(d.chol.U)[i], 'U', 0),
                    ),
                    reshape(y[i, :], 1, size(y, 2))
                ) for i in 1:length(blocks(d.Σ))
            ]
        )
    else
        return sum(
            [
                logpdf(
                    Gaussian(
                        reshape(d.μ[i, :], 1, size(d.μ, 2)),
                        blocks(d.Σ)[i]
                    ),
                    reshape(y[i, :], 1, size(y, 2))
                ) for i in 1:length(blocks(d.Σ))
            ]
        )
    end
end

"""
    function reglogpdf(
        reg::Function,
        gp::GP,
        x,
        y::AbstractArray{<:Real},
        params::Vector{G}
    ) where {G <: Real}

Compute the log-pdf of `gp` over inputs `x` and outputs `y`, using `reg` as a regulariser
and `params` as the parameters of `gp`. Note that `reg` *must* be of type signature
`reg(::GP, x, ::::AbstractArray{<:Real})`. The value of `reg` will be *subtracted* from the
log-pdf.
"""
@unionise function reglogpdf(
    reg::Function,
    gp::GP,
    x,
    y::AbstractArray{<:Real},
    params::Vector{G}
) where {G <: Real}
    # update kernels with new parameters.
    # if we want to update the means as well, we should overload this.
    ngp = GP(gp.m, set(gp.k, params))
    return logpdf(ngp::GP, x, y) - reg(ngp, x, y)
end

@unionise function Distributions.logpdf(
    gp::GP,
    x,
    y::AbstractArray{<:Real},
    params::Vector{G}
) where {G <: Real}
    return reglogpdf((a, b, c) -> 0.0, gp, x, y, params)
end

# We need a different one now for the OLMM that ensures that H is updated properly.
@unionise function reglogpdf(
    reg::Function,
    gp::GP{K, M},
    x,
    y::AbstractArray{<:Real},
    params::Vector{G}
) where {K <: OLMMKernel, M <: Mean, G <: Real}
    # This has the updated H, but the old U. H might (and usually will) not be of the form
    # H = U. S.
    ngp = GP(gp.m, set(gp.k, params))
    isa(ngp.k.H, Fixed) || _constrain_H!(ngp)
    return logpdf(ngp::GP, x, y::AbstractArray) - reg(ngp, x, y)
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
    gp::GP{K, M},
    x,
    y::AbstractMatrix{<:Real}
) where {K <: OLMMKernel, M <: Mean}

    n = size(x, 1)
    p = unwrap(gp.k.p)
    m = unwrap(gp.k.m)
    σ² = Ones(p) .* unwrap(gp.k.σ²)
    H = float(unwrap(gp.k.H)) # Prevents Nabla from breaking in case H has Ints.
    D = unwrap(gp.k.D)
    S_sqrt = unwrap(gp.k.S_sqrt)
    isa(gp.k.ks, Kernel) && !isa(D, Vector) && S_sqrt ≈ ones(m) && return optlogpdf(gp, x, y) # TODO: voltar aqui


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
        proj_noise = (unwrap(gp.k.σ²)/(S_sqrt[i])^2 + D[i])
        Σlk = gp.k.ks[i](x)
        glk = Gaussian(Zeros(n), proj_noise * Eye(n) + Σlk)
        yls = yl[:, i]
        lpdf += logpdf(glk, yls) + 0.5 * (n * log(2π * proj_noise) + yls' * yls / proj_noise)
    end
    return lpdf
end

@unionise function Distributions.logpdf(gp::GP, x, y::AbstractArray{<:Real})
    return logpdf(gp(x), y)
end

"""
    mle_obj(gp::GP, x, y::AbstractArray) -> Function

Objective function that, when minimised, yields maximum likelihood of observations `y` for
a `gp` evaluated at points `x`. Returns a function of the `GP` parameters. Use this for
maximum likelihood estimates.
"""
@unionise function mle_obj(gp::GP, x, y::AbstractArray{<:Real})
    return function f(params)
        return -logpdf(gp::GP, x, y, params)
    end
end

"""
    map_obj(reg::Function, gp::GP, x, y::AbstractArray) -> Function

Objective function that, when minimised, yields maximum likelihood of observations `y` for
a `gp` evaluated at points `x`, regularised by `reg`. Returns a function of the `GP`
parameters.  Note that `reg` *must* be of type signature
`reg(::GP, x, ::::AbstractArray{<:Real})`. Use this for maximum a posteriori estimates.
"""
@unionise function map_obj(reg::Function, gp::GP, x, y::AbstractArray{<:Real})
    return function f(params)
        return -reglogpdf(reg::Function, gp::GP, x, y, params)
    end
end

@unionise function map_obj(reg::Function)
    return function f(gp::GP, x, y)
        return map_obj(reg::Function, gp::GP, x, y)
    end
end

"""
    titsiasELBO(gp::GP, x, y::AbstractArray{<:Real})

Compute the lower bound for the posterior logpdf under Titsias' approach. See:
"Variational Learning of Inducing Variables in Sparse Gaussian Processes"
"""
@unionise function titsiasELBO(gp::GP, x, y::AbstractVector{<:Real})
    Xm = unwrap(gp.k.Xm)
    k = gp.k.k
    m = gp.m
    σ² = unwrap(gp.k.σ²)
    num_m = unwrap(gp.k.n)
    n = size(x, 1)
    # Compute first term
    Kmm = k(Xm, Xm)
    Kmn = k(Xm, x)
    Umm = cholesky(Kmm + _EPSILON_^2 * Eye(num_m)).U
    T = Umm' \ Kmn
    P = Eye(num_m) + (T * T') ./ σ²
    Up = cholesky(P).U
    L = (Up * Umm)'
    # The implementation above should be mathematically equivalent to the one below, but
    # numerically more stable.
    # L = cholesky(Kmm + Kmn * Kmn' ./ σ² + _EPSILON_^2 * Eye(num_m)).L
    log_dets = -sum(log, diag(Umm)) + sum(log, diag(L))
    μ = y .- m(x)
    Z = L \ (Kmn * μ)
    log_N = -0.5 * (n * log(2π * σ²) + 2 * log_dets + ((μ' * μ) - (Z' * Z) / σ²) / σ²)
    # Compute K̅
    return log_N - (sum(var(k, x)) - sum(w -> w^2, T)) / (2 * σ²)
end

# This is here simply for reference, as it is more readable.
@unionise function slowtitsiasELBO(gp::GP, x, y::AbstractArray{<:Real})
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
    return log_N - (2 * σ²)^(-1) * (sum(var(k, x)) - tr(Qnn))
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
        proj_noise = unwrap(gp.k.k.σ²) / S_sqrt[i]^2 + D[i]
        pσ² = sσ² + proj_noise
        # Here we compute the sparse GP contributions
        Kmm = gp.k.k.ks[i](Xm, Xm)
        Kmn = gp.k.k.ks[i](Xm, x)
        Umm = cholesky(Kmm + _EPSILON_^2 * Eye(num_m)).U
        T = Umm' \ Kmn
        P = Eye(num_m) + (T * T') ./ pσ²
        Up = cholesky(P).U
        L = (Up * Umm)'
        # The implementation above should be mathematically equivalent to the one below, but
        # numerically more stable.
        # L = cholesky(Kmm + Kmn * Kmn' ./ σ² + _EPSILON_^2 * Eye(num_m)).L
        log_dets = -sum(log, diag(Umm)) + sum(log, diag(L))
        μ = yl[:, i]
        Z = L \ (Kmn * μ)
        log_N = -0.5 * (n * log(2π * pσ²) + 2 * log_dets + (μ' * μ) / pσ² - (Z' * Z) / (pσ²)^2)
        slpdf = log_N - (sum(var(gp.k.k.ks[i], x)) - sum(w -> w^2, T)) / (2 * sσ²)
        lpdf += slpdf + 0.5 * (n * log(2π * proj_noise) + μ' * μ / proj_noise)
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
    ngp = GP(gp.m, sk)
    return function f(params)
        return -titsiasELBO(ngp, x, y, params)
    end
end
# TODO: A method that let's specify only the number of inducing points

"""
    unconstrained_markowitz(gp::GP, x; α::Real=1)

Perform unconstrained mean-variance Markowitz optimisation, using risk aversion parameter
`α`, for input `x`. Returns the optimal weigths. This assumes a single timestamp is being
provided.

This is a simplified version of our Markowitz PO since it does not include the balance
constraint (total net volume equal 0), the maximum volume constraint (total absolute
volume less than M) or the maximum nodal volume constraint (total absolute volume on any
node less than r*M), for some given values r and M.
"""
@unionise function _unconstrained_markowitz(gp::GP, x; α::Real=1)
    α <= 0 && throw(ArgumentError("Risk aversion parameter must be positive, received $α"))
    return 1/(2α) * gp.k(x) \ gp.m(x)'
end

@unionise function _unconstrained_markowitz(gp::GP{<:OLMMKernel}, x; α::Real=1)
    α <= 0 && throw(ArgumentError("Risk aversion parameter must be positive, received $α"))

    H = unwrap(gp.k.H)
    σ² = unwrap(gp.k.σ²)
    D = unwrap(gp.k.D)
    m = unwrap(gp.k.m)
    D = isa(D, Float64) ? fill(D, m) : D
    p = unwrap(gp.k.p)

    # The reshape and the vcat are tricks to make Nabla work.
    K_ = Diagonal(vcat([reshape(k(x), 1) for k in gp.k.ks]...))
    Σ_ = H * K_ * H'
    Σ = Σ_ + σ² * Eye(p) + H * (D .* Eye(m)) * H'

    return 1/(2α) * Σ \ gp.m(x)'
end

# TODO: implement an optimised version for the LMM as well.
# https://gitlab.invenia.ca/research/GPForecasting.jl/issues/55

"""
    norm_expected_return(gp::GP, x, y; α::Real=1)

Return the normalised expected return for a forecast distribution `gp(x)` and actuals `y`,
using an unconstrained Markowitz solution for the weights, with risk aversion parameter `α`.

The normalisation means that the weight vector has unit norm, i.e., this is insensitive to
uniform scalings of the volumes.

If `x` represents a single timestamp, `y` should be a vector. If `x` represents several
timestamps, `y` should be a matrix with the number of rows equal to the number of timestamps.

"""
@unionise function norm_expected_return(gp::GP, x, y::Vector{<:Real}; α::Real=1)
    vols = _unconstrained_markowitz(gp, x, α=α)
    return dot(vols ./ sqrt(dot(vols, vols) + 1e-15), y)
end

@unionise function norm_expected_return(gp::GP, x, y::Matrix{<:Real}; α::Real=1)
    # We don't want this breaking if we send a single timestamp as a row matrix.
    size(y, 1) == 1 && return norm_expected_return(gp, x, dropdims(y, dims=1), α=α)
    size(x, 1) != size(y, 1) && throw(ArgumentError("x and y must have same number of rows"))
    if isa(x, DataFrame)
        return sum(
            [
                norm_expected_return(gp, DataFrame(x[i, :]), y[i, :], α=α)
                for i in 1:size(x, 1)
            ]
        )
    else
        return sum(
            [norm_expected_return(gp, x[i, :], y[i, :], α=α) for i in 1:size(x, 1)]
        )
    end
end

"""
    norm_expected_posterior_return(
        gp::GP,
        xc,
        xt,
        yc::AbstractArray{<:Real},
        yt::AbstractArray{<:Real},
        params;
        α::Real=1,
    )

Compute expected return of the `gp` conditioned on `xc` and `yc` over the pair (`xt`, `yt`).
It is important to have (`xc`, `yc`) disjoint with (`xt`, `yt`) because the posterior usually
closely reproduces the conditioned data.
"""
@unionise function norm_expected_posterior_return(
    gp::GP,
    xc,
    xt,
    yc::AbstractArray{<:Real},
    yt::AbstractArray{<:Real},
    params;
    α::Real=1,
)
    ngp = GP(gp.m, set(gp.k, params))
    # Build posterior
    pos = condition(ngp, xc, yc)
    return norm_expected_return(pos, xt, yt, α=α)
end

@unionise function norm_expected_posterior_return(
    gp::GP{<:OLMMKernel},
    xc,
    xt,
    yc::AbstractArray{<:Real},
    yt::AbstractArray{<:Real},
    params;
    α::Real=1,
)
    # This has the updated H, but the old U. H might (and usually will) not be of the form
    # H = U. S.
    ngp = GP(gp.m, set(gp.k, params))
    isa(ngp.k.H, Fixed) || _constrain_H!(ngp)
    # Build posterior
    pos = condition(ngp, xc, yc)
    return norm_expected_return(pos, xt, yt, α=α)
end

"""
    norm_expected_posterior_return_obj(gp::GP, x, y::AbstractArray{<:Real}; α::Real=1)

Objective function that, when minimised, yields maximum expected return for a forecast
distribution `gp(x)` and actuals `y`, using an unconstrained Markowitz solution for the
weights, with risk aversion parameter `α`. The expected return is computed independently for each
timestamp.
"""
@unionise function norm_expected_posterior_return_obj(
    gp::GP,
    xc,
    xt,
    yc::AbstractArray{<:Real},
    yt::AbstractArray{<:Real};
    α::Real=1,
)
    return function f(params)
        return -norm_expected_posterior_return(gp, xc, xt, yc, yt, params; α=α)
    end
end

"""
    norm_expected_return_balanced(gp::GP, x, y; α::Real=1, λ::Real=100)

Return the normalised expected return for a forecast distribution `gp(x)` and actuals `y`,
using an unconstrained Markowitz solution for the weights, with risk aversion parameter `α`
and penalising the result by `λ` times the net volume. The penalty term encourages balanced
volumes (see `GPForecasting._unconstrained_markowitz`).

The normalisation means that the weight vector has unit norm, i.e., this is insensitive to
uniform scalings of the volumes.

If `x` represents a single timestamp, `y` should be a vector. If `x` represents several
timestamps, `y` should be a matrix with the number of rows equal to the number of timestamps.
"""
@unionise function norm_expected_return_balanced(
    gp::GP,
    x,
    y::Vector{<:Real};
    α::Real=1,
    λ::Real=100,
)
    vols = _unconstrained_markowitz(gp, x, α=α)
    profit = dot(vols ./ sqrt(dot(vols, vols) + 1e-15), y)
    regulariser = λ * abs(sum(vols ./ sqrt(dot(vols, vols) + 1e-15)))
    return profit - regulariser
end

@unionise function norm_expected_return_balanced(
    gp::GP,
    x,
    y::Matrix{<:Real};
    α::Real=1,
    λ::Real=100,
)
    # We don't want this breaking if we send a single timestamp as a row matrix.
    if size(y, 1) == 1
        return norm_expected_return_balanced(gp, x, dropdims(y, dims=1), α=α, λ=λ)
    end
    size(x, 1) != size(y, 1) && throw(ArgumentError("x and y must have same number of rows"))
    if isa(x, DataFrame)
        return sum(
            [
                norm_expected_return_balanced(gp, DataFrame(x[i, :]), y[i, :], α=α, λ=λ)
                for i in 1:size(x, 1)
            ]
        )
    else
        return sum(
            [
                norm_expected_return_balanced(gp, x[i, :], y[i, :], α=α, λ=λ)
                for i in 1:size(x, 1)
            ]
        )
    end
end

"""
    norm_expected_posterior_return_balanced(
        gp::GP,
        xc,
        xt,
        yc::AbstractArray{<:Real},
        yt::AbstractArray{<:Real},
        params;
        α::Real=1,
        λ::Real=100,
    )

Compute norm_expected_return_balanced of the `gp` conditioned on `xc` and `yc` over the pair
(`xt`, `yt`). It is important to have (`xc`, `yc`) disjoint with (`xt`, `yt`) because the
posterior usually closely reproduces the conditioned data.
"""
@unionise function norm_expected_posterior_return_balanced(
    gp::GP,
    xc,
    xt,
    yc::AbstractArray{<:Real},
    yt::AbstractArray{<:Real},
    params;
    α::Real=1,
    λ::Real=100,
)
    ngp = GP(gp.m, set(gp.k, params))
    # Build posterior
    pos = condition(ngp, xc, yc)
    return norm_expected_return_balanced(pos, xt, yt, α=α, λ=λ)
end

@unionise function norm_expected_posterior_return_balanced(
    gp::GP{<:OLMMKernel},
    xc,
    xt,
    yc::AbstractArray{<:Real},
    yt::AbstractArray{<:Real},
    params;
    α::Real=1,
    λ::Real=100,
)
    # This has the updated H, but the old U. H might (and usually will) not be of the form
    # H = U * S
    ngp = GP(gp.m, set(gp.k, params))
    isa(ngp.k.H, Fixed) || _constrain_H!(ngp)
    # Build posterior
    pos = condition(ngp, xc, yc)
    return norm_expected_return_balanced(pos, xt, yt, α=α, λ=λ)
end

"""
    norm_expected_posterior_return_balanced_obj(
        gp::GP,
        xc,
        xt,
        yc::AbstractArray{<:Real},
        yt::AbstractArray{<:Real};
        α::Real=1,
        λ::Real=100,
    )

Objective function that, when minimised, yields maximum expected return for a forecast
distribution `gp(x)` and actuals `y`, using an unconstrained Markowitz solution for the
weights, with risk aversion parameter `α`. The expected return is computed independently for
each timestamp.
"""
@unionise function norm_expected_posterior_return_balanced_obj( # Verbose as hell.
    gp::GP,
    xc,
    xt,
    yc::AbstractArray{<:Real},
    yt::AbstractArray{<:Real};
    α::Real=1,
    λ::Real=100,
)
    return function f(params)
        return -norm_expected_posterior_return_balanced(gp, xc, xt, yc, yt, params; α=α, λ=λ)
    end
end
