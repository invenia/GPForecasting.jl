"""
    GP{K<:Kernel, M<:Mean} <: Process

Gaussian process.

# Fields:
- `m::Mean`: Mean
- `k::Kernel`: Kernel

# Constructors:
    GP(m::Mean, k::Kernel)

    GP(n::Real, k::Kernel)

Return GP with constant mean `n`.

    GP(k::Kernel)

Return GP with zero mean.
"""
mutable struct GP{K<:Kernel, M<:Mean} <: Process
    m::M
    k::K
end

function GP(n::Real, k::Kernel)
    return GP(
        isMulti(k) ?
        MultiMean([ConstantMean(n) for i in 1:size(k, 1)]) :
        ConstantMean(n),
        k,
    )
end
GP(k::Kernel) = GP(0.0, k)

Base.:(==)(a::GP, b::GP) = GPForecasting.get(a) == GPForecasting.get(b)
Base.:*(gp::GP, x) = GP(x * gp.m, x * gp.k * x')
Base.:*(n, gp::GP) = gp * n
Base.:+(g1::GP, g2::GP) = GP(g1.m + g2.m, g1.k + g2.k)
Base.:-(g1::GP, g2::GP) = GP(g1.m - g2.m, g1.k + g2.k)

"""
    condition(gp::GP, x, y) -> GP

Condition a Gaussian process, i.e., returns the stochastic process that corresponds to the
original one conditioned on the observation of values `y` over inputs `x`. The output GP
is the posterior process corresponding to the prior updated with the observations `y`.

# Arguments
- `gp::GP`: The process to condition
- `x`: The input locations of the observations
- `y`: The values of the observations

# Returns
- `GP`: The posterior process
"""
function condition(gp::GP, x, y::AbstractArray{<:Real})
    # This call to Hermitian should not be necessary, but numerical noise has been making
    # the cholesky fail. TODO: investigate the source of the noise. It only happens if we
    # learn `H` in the OLMM, but all the constraints are respected.
    K = Hermitian(gp.k(x))
    U = cholesky(K + _EPSILON_ * Eye(K)).U
    m = PosteriorMean(gp.k, gp.m, x, U, y)
    k = PosteriorKernel(gp.k, x, U)
    return GP(m, k)
end
# TODO: Implement conditioning over the extended input space.

function condition(
    gp::GP{T, G},
    x,
    y::AbstractMatrix{<:Real}
) where {T <: LMMKernel, G <: Mean} # Assuming always zero mean here. We should properly dispatch later
    m = unwrap(gp.k.m)
    n = size(x, 1)
    p = unwrap(gp.k.p)
    ???? = unwrap(gp.k.????)
    ???? = isa(????, Float64) ? ones(p, 1) * ???? : reshape(????, p, 1)
    H = unwrap(gp.k.H)

# NOTE: Our LMM notes have all been derived assuming a different convention for inputs
# and outputs. Thus, inside this function we will convert them to old conventions and
# convert back before returning the output.
    yt = y'

    Us = [cholesky(k(x, x) + _EPSILON_ * Eye(n)).U for k in gp.k.ks]
    yi???? = yt ./ ????
    Hi??y = reshape(H' * yi????, n * m, 1)
    UQ = sum_kron_J_ut(m, Us...)
    M = cholesky(Symmetric(eye_sum_kron_M_ut(transpose(H) * (H ./ ????), Us...))).U
    Z = UQ' / M

    m = LMMPosMean(gp.k, x, Z, y) # NOTE: assuming here zero prior mean for the LMM.
    k = LMMPosKernel(gp.k, x, Z)
    return GP(m, k)
end

function condition(
    gp::GP{T, G},
    x,
    y::AbstractMatrix{<:Real}
) where {T <: LMMPosKernel, G <: LMMPosMean}
# This is a way of getting around the issue of re-deriving and re-optimisng the
# posterior kernel and mean expressions. Should not be the most efficient approach
# possible, but is a solution.
    old_x = unwrap(gp.m.x)
    old_y = unwrap(gp.m.y)
    new_x = vcat(old_x, x)
    new_y = vcat(old_y, y)
    return condition(GP(gp.k.k), new_x, new_y)
end

function optcondition(
    gp::GP{T, G},
    x,
    y::AbstractMatrix{<:Real}
) where {T <: OLMMKernel, G <: Mean}
    P = unwrap(gp.k.P)
    m = unwrap(gp.k.m)
    ???? = unwrap(gp.k.????)
    d = unwrap(gp.k.D)
    D = fill(d, m)
    yp = y * P'
    # condition gp.k.ks on y
    kx = gp.k.ks(x)
    K = kx + (???? + d) * I

    U = cholesky(Hermitian(K + _EPSILON_^2 * Eye(K))).U
    ms = [PosteriorMean(gp.k.ks, ZeroMean(), x, U, yp[:, i]) for i in 1:m]
    ks = PosteriorKernel(gp.k.ks, x, U)
    # create the posterior mean (storing y rather than yp is useful for downstream analysis)
    pos_m = OLMMPosMean(gp.k, ms, x, y)
    # create another OLMMKernel
    pos_k = _unsafe_OLMMKernel(
        gp.k.m,
        gp.k.p,
        gp.k.????,
        gp.k.D,
        gp.k.H,
        gp.k.P,
        gp.k.U,
        gp.k.S_sqrt,
        ks
    )
    return GP(pos_m, pos_k)
end

function condition(
    gp::GP{T, G},
    x,
    y::AbstractMatrix{<:Real}
) where {T <: OLMMKernel, G <: Mean}
    # project y
    P = unwrap(gp.k.P)
    m = unwrap(gp.k.m)
    ???? = unwrap(gp.k.????)
    D = unwrap(gp.k.D)
    S_sqrt = unwrap(gp.k.S_sqrt)
    isa(gp.k.ks, Kernel) && !isa(D, Vector) && S_sqrt ??? ones(m) && return optcondition(gp, x, y)
    # More Nabla stuff
    D = size(D) == () ? D * ones(m) : D
    yp = y * P'
    # condition gp.k.ks on y
    Ks = []
    # Nabla's Branch and Leaf are not Iterable, so can't do for (k, s, d) in zip(gp.k.ks, S_sqrt, D)
    for i in 1:length(gp.k.ks)
        kx = gp.k.ks[i](x)
        # This is clumsy, but it is the most robust way of getting the right size that has
        # no corner cases, and we can't call Eye(kx) because Nabla
        n = size(kx, 1)
        K = kx + (????/S_sqrt[i]^2 + D[i]) * Eye(n)
        push!(Ks, K)
    end
    Us = [cholesky(Symmetric(K + _EPSILON_^2 * Eye(size(K, 1)))).U for K in Ks]
    ms = [PosteriorMean(k, ZeroMean(), x, U, yp[:, i]) for (U, k, i) in zip(Us, gp.k.ks, 1:m)]
    ks = [PosteriorKernel(k, x, U) for (U, k) in zip(Us, gp.k.ks)]
    # create the posterior mean (storing y rather than yp is useful for downstream analysis)
    pos_m = OLMMPosMean(gp.k, ms, x, y)
    # create another OLMMKernel
    pos_k = _unsafe_OLMMKernel(
        gp.k.m,
        gp.k.p,
        gp.k.????,
        gp.k.D,
        gp.k.H,
        gp.k.P,
        gp.k.U,
        gp.k.S_sqrt,
        ks
    )
    return GP(pos_m, pos_k)
end

function condition(
    gp::GP{T, G},
    x,
    y::AbstractMatrix{<:Real}
) where {T <: LSOLMMKernel, G <: Mean}
    pgp = condition(GP(gp.m, gp.k.olmm), x, y)
    pos_m = pgp.m # Posterior mean
    pos_olmm = pgp.k # Posterior OLMM
    k = gp.k # Prior LSOLMM
    return GP(pos_m, _unsafe_LSOLMMKernel(k.Hk, k.lat_pos, k.out_pos, pos_olmm))
end

"""
    condition_sparse(gp::GP, x, Xm, y::AbstractArray{<:Real}, ????)

Condition GP on observed data. This should satisfy both "Variational Learning of Inducing
Variables in Sparse Gaussian Processes" and "Bayesian Gaussian Process Models:
PAC-Bayesian Generalisation Error Bounds and Sparse Approximations.". For other approaches,
some form of dispatching would be needed.
"""
function condition_sparse(gp::GP, x, Xm, y::AbstractArray{<:Real}, ????)
    xm = unwrap(Xm)
    pos_m, pos_k = _condition_sparse(gp.k, gp.m, x, xm, y, ????)
    return GP(pos_m, pos_k)
end

function _condition_sparse(k::Kernel, m::Mean, x, Xm, y::AbstractArray{<:Real}, ????; )
    # Compute the relevant Choleskys
    Kmm = k(Xm, Xm)
    Umm = cholesky(Kmm + _EPSILON_^2 * I).U
    Knm = k(x, Xm)
    T = Umm' \ Knm'
    P = I + (1/unwrap(????)) .* (T * T')
    Up = cholesky(P + _EPSILON_^2 * I).U
    Uz = Up * Umm
    # The implementation above should be mathematically equivalent to the one below, but
    # numerically more stable.
    # Z = Kmm + (1/unwrap(????)) .* Knm' * Knm # Z = inv(??)
    # Uz = Nabla.chol(Z + _EPSILON_^2 * I)

    # Build posterior TitsiasPosteriorKernel and TitsiasPosteriorMean
    pos_m = TitsiasPosteriorMean(k, m, x, Xm, Uz, ????, y)
    pos_k = TitsiasPosteriorKernel(k, Xm, Uz, Umm, ????)
    return pos_m, pos_k
end

function condition_sparse(gp::GP{<:OLMMKernel, <:Mean}, x, Xm, y::AbstractArray{<:Real}, s????)
    # project y
    P = unwrap(gp.k.P)
    m = unwrap(gp.k.m)
    ???? = unwrap(gp.k.????)
    D = unwrap(gp.k.D)
    S_sqrt = unwrap(gp.k.S_sqrt)
    D = isa(D, Float64) ? fill(D, m) : D
    xm = unwrap(Xm)
    yp = y * P'
    # sparse condition gp.k.ks on y
    pos_ks = Vector{TitsiasPosteriorKernel}(undef, m)
    pos_ms = Vector{TitsiasPosteriorMean}(undef, m)
    for (k, s, d, i) in zip(gp.k.ks, S_sqrt, D, collect(1:m))
        # This is a temporary kernel meant to reuse the condition_sparse code
        tk = k + (????/s^2 + d) * DiagonalKernel()
        pm, pk = _condition_sparse(tk, ZeroMean(), x, xm, yp[:, i], s????)
        # Now we move back to the proper kernels that should be stored.
        pk.k = k
        pm.k = k
        pos_ks[i] = pk
        pos_ms[i] = pm
    end
    pos_m = OLMMPosMean(gp.k, pos_ms, x, yp)
    # create another OLMMKernel
    pos_k = _unsafe_OLMMKernel(
        gp.k.m,
        gp.k.p,
        gp.k.????,
        gp.k.D,
        gp.k.H,
        gp.k.P,
        gp.k.U,
        gp.k.S_sqrt,
        pos_ks
    )
    return GP(pos_m, pos_k)
end

(p::GP)(x; hourly=false) = Gaussian(p.m(x), hourly ? hourly_cov(p.k, x) : p.k(x))
function (p::GP{K, M})(x::Input; hourly=false) where {K <: NoiseKernel, M <: Mean}
    return Gaussian(p.m(x.val), hourly ? hourly_cov(p.k, x) : p.k(x))
end
function (p::GP{K, M})(x::Vector{Input}) where {K <: NoiseKernel, M <: Mean}
    cx = vcat([c.val for c in x]...)
    return Gaussian(p.m(cx), p.k(x))
end
function (p::GP{K, L})(x) where {K <: NoiseKernel, L <: Mean}
    # Here we will work in the extended input space
    M = p.m(x)
    return Gaussian(stack([M, M]), p.k(x))
end
function (p::GP{K, L})(x) where {K <: NoiseKernel, L <: MultiMean}
    return Gaussian(p.m(x), p.k(x))
end

Distributions.MvNormal(gp::GP, x) = MvNormal(collect(vec(gp.m(x)[:, :]')), Matrix(Hermitian(gp.k(x))))
function Distributions.MvNormal(p::GP{K, M}, x::Input) where {K <: NoiseKernel, M <: Mean}
    return MvNormal(collect(vec(p.m(x.val)[:, :]')), Matrix(Hermitian(p.k(x))))
end
function Distributions.MvNormal(p::GP{K, L}, x) where {K <: NoiseKernel, L <: Mean}
    # Here we will work in the extended input space
    M = p.m(x)
    return MvNormal(collect(vec(stack([M, M])[:, :]')), Matrix(Hermitian(p.k(x))))
end
function Distributions.MvNormal(p::GP{K, L}, x) where {K <: NoiseKernel, L <: MultiMean}
    return MvNormal(collect(vec(p.m(x)[:, :]')), Matrix(Hermitian(p.k(x))))
end

"""
    credible_interval(p::GP, x) -> Tuple{Vector{<:Real}, Vector{<:Real}, Vector{<:Real}}

Get the mean and upper and lower central 95%???credible bounds at certain inputs `x`.

# Arguments
- `p::GP`: The GP to compute the central 95%???credible intervals for
- `x`: The points at which to compute the central 95%???credible intervals

# Returns
- `Vector{<:Real}`: Means
- `Vector{<:Real}`: Lower central 95%???credible bound
- `Vector{<:Real}`: Upper central 95%???credible bound
"""
function _noisy_ci_(p::GP, x)
    # Here we will work in the extended input space
    M = p.m(x)
    ?? = stack([M, M])
    ?? = sqrt.(max.(var(p.k, x), 0))
    return ??, ?? .- 2 .* ??, ?? .+ 2 .* ??
end
function _ci_(p::GP, x)
    ?? = sqrt.(max.(var(p.k, x), 0))
    ?? = p.m(x)
    return ??, ?? .- 2 .* ??, ?? .+ 2 .* ??
end
function credible_interval(p::GP, x)
    # TODO: Should directly obtain the marginal variances whenever this is supported.
    isa(p.k, PosteriorKernel) && isa(p.k.k, NoiseKernel) && return _noisy_ci_(p, x)
    return _ci_(p, x)
end
function credible_interval(p::GP{K, L}, x::Input) where {K <: NoiseKernel, L <: Mean}
    return _ci_(p, x)
end
function credible_interval(p::GP{K, L}, x::Input) where {K <: PosteriorKernel, L <: Mean}
    return _ci_(p, x)
end
function credible_interval(p::GP{K, L}, x::Vector{Input}) where {K <: NoiseKernel, L <: Mean}
    return _ci_(p, x)
end
function credible_interval(p::GP{K, L}, x::Vector{Input}) where {K <: PosteriorKernel, L <: Mean}
    return _ci_(p, x)
end
function credible_interval(p::GP{K, L}, x) where {K <: NoiseKernel, L <: Mean}
    return _noisy_ci_(p, x)
end
