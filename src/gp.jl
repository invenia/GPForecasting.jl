export Process, GP, condition, credible_interval

function GP(n::Real, k::Kernel)
    return GP(
        isMulti(k) ?
        MultiMean([ConstantMean(n) for i in 1:size(k, 1)]) :
        ConstantMean(n),
        k,
    )
end
GP(k::Kernel) = GP(0.0, k)

(*)(gp::GP, x) = GP(x * gp.m, x * gp.k * x')
(*)(n, gp::GP) = (*)(gp::GP, n)
(+)(g1::GP, g2::GP) = GP(g1.m + g2.m, g1.k + g2.k)
(-)(g1::GP, g2::GP) = GP(g1.m - g2.m, g1.k + g2.k)

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
function condition(gp::GP, x, y)
    K = gp.k(x)
    U = chol(K + _EPSILON_ * eye(K))
    m = PosteriorMean(gp.k, gp.m, x, U, y)
    k = PosteriorKernel(gp.k, x, U)
    return GP(m, k)
end
# TODO: Implement conditioning over the extended input space.

function condition(
    gp::GP{T, G},
    x,
    y
) where {T <: LMMKernel, G <: Mean} # Assuming always zero mean here. We should properly dispatch later
    m = unwrap(gp.k.m)
    n = size(x, 1)
    p = unwrap(gp.k.p)
    σ² = unwrap(gp.k.σ²)
    σ² = isa(σ², Float64) ? ones(p, 1) * σ² : reshape(σ², p, 1)
    H = unwrap(gp.k.H)

# NOTE: Our LMM notes have all been derived assuming a different convention for inputs
# and outputs. Thus, inside this function we will convert them to old conventions and
# convert back before returning the output.
    yt = y'

    Us = [chol(k(x, x) + _EPSILON_ * eye(n)) for k in gp.k.ks]
    yiσ² = yt ./ σ²
    HiΛy = reshape(H' * yiσ², n * m, 1)
    UQ = sum_kron_J_ut(m, Us...)
    M = chol(Symmetric(eye_sum_kron_M_ut(At_mul_B(H, H ./ σ²), Us...)))
    Z = UQ' / M

    m = LMMPosMean(gp.k, x, Z, y) # NOTE: assuming here zero prior mean for the LMM.
    k = LMMPosKernel(gp.k, x, Z)
    return GP(m, k)
end

function condition(
    gp::GP{T, G},
    x,
    y
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

function condition(
    gp::GP{T, G},
    x,
    y
) where {T <: OLMMKernel, G <: Mean}
    # project y
    P = unwrap(gp.k.P)
    m = unwrap(gp.k.m)
    σ² = unwrap(gp.k.σ²)
    D = unwrap(gp.k.D)
    D = isa(D, Float64) ? fill(D, m) : D
    S_sqrt = unwrap(gp.k.S_sqrt)
    yp = y * P'
    # condition gp.k.ks on y
    Ks = []
    for (k, s, d) in zip(gp.k.ks, S_sqrt, D)
        kx = k(x)
        push!(Ks, kx + (σ²/s^2 + d) * eye(kx))
    end
    Us = [chol(Hermitian(K + _EPSILON_^2 * eye(K))) for K in Ks]
    ms = [PosteriorMean(k, ZeroMean(), x, U, yp[:, i]) for (U, k, i) in zip(Us, gp.k.ks, 1:m)]
    ks = [PosteriorKernel(k, x, U) for (U, k) in zip(Us, gp.k.ks)]
    # create the posterior mean
    pos_m = OLMMPosMean(gp.k, ms, x, yp)
    # create another OLMMKernel
    pos_k = _unsafe_OLMMKernel(
        gp.k.m,
        gp.k.p,
        gp.k.σ²,
        gp.k.D,
        gp.k.H,
        gp.k.P,
        gp.k.U,
        gp.k.S_sqrt,
        ks
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

MvNormal(gp::GP, x) = MvNormal(vec(gp.m(x)[:, :]'), Matrix(Hermitian(gp.k(x))))
function MvNormal(p::GP{K, M}, x::Input) where {K <: NoiseKernel, M <: Mean}
    return MvNormal(vec(p.m(x.val)[:, :]'), Matrix(Hermitian(p.k(x))))
end
function MvNormal(p::GP{K, L}, x) where {K <: NoiseKernel, L <: Mean}
    # Here we will work in the extended input space
    M = p.m(x)
    return MvNormal(vec(stack([M, M])[:, :]'), Matrix(Hermitian(p.k(x))))
end
function MvNormal(p::GP{K, L}, x) where {K <: NoiseKernel, L <: MultiMean}
    return MvNormal(vec(p.m(x)[:, :]'), Matrix(Hermitian(p.k(x))))
end

"""
    credible_interval(p::GP, x) -> Tuple{Vector{<:Real}, Vector{<:Real}, Vector{<:Real}}

Get the mean and upper and lower central 95%–credible bounds at certain inputs `x`.

# Arguments
- `p::GP`: The GP to compute the central 95%–credible intervals for
- `x`: The points at which to compute the central 95%–credible intervals

# Returns
- `Vector{<:Real}`: Means
- `Vector{<:Real}`: Lower central 95%–credible bound
- `Vector{<:Real}`: Upper central 95%–credible bound
"""
function _noisy_credible_interval(p::GP, x)
    # Here we will work in the extended input space
    M = p.m(x)
    μ = stack([M, M])
    σ = sqrt.(max.(var(p.k, x), 0))
    return μ, μ .- 2 .* σ, μ .+ 2 .* σ
end
function credible_interval(p::GP, x)
    # TODO: Should directly obtain the marginal variances whenever this is supported.
    σ = sqrt.(max.(var(p.k, x), 0))
    μ = p.m(x)
    return μ, μ .- 2 .* σ, μ .+ 2 .* σ
end
function _ci_n_i_(p::GP, x::Input)
    σ = sqrt.(max.(var(p.k, x), 0))
    μ = p.m(x.val)
    return μ, μ .- 2 .* σ, μ .+ 2 .* σ
end
function credible_interval(p::GP{K, L}, x::Input) where {K <: NoiseKernel, L <: Mean}
    return _ci_n_i_(p, x)
end
function credible_interval(p::GP{K, L}, x::Input) where {K <: PosteriorKernel, L <: Mean}
    return _ci_n_i_(p, x)
end
function _ci_n_vi_(p::GP, x::Vector{Input})
    σ = sqrt.(max.(var(p.k, x), 0))
    cx = vcat([c.val for c in x]...)
    μ = p.m(cx)
    return μ, μ .- 2 .* σ, μ .+ 2 .* σ
end
function credible_interval(p::GP{K, L}, x::Vector{Input}) where {K <: NoiseKernel, L <: Mean}
    return _ci_n_vi_(p, x)
end
function credible_interval(p::GP{K, L}, x::Vector{Input}) where {K <: PosteriorKernel, L <: Mean}
    return _ci_n_vi_(p, x)
end
function credible_interval(p::GP{K, L}, x) where {K <: NoiseKernel, L <: Mean}
    return _noisy_credible_interval(p, x)
end
function credible_interval(p::GP{MultiOutputKernel, MultiOutputMean}, x)
    μ = p.m(x)
    σ = sqrt.(max.(var(p.k, x), 0))
    return μ, μ .- 2 .* σ, μ .+ 2 .* σ
end
function credible_interval(p::GP{LMMPosKernel, LMMPosMean}, x)
    μ = p.m(x)
    σ = sqrt.(max.(var(p.k, x), 0))
    return μ, μ .- 2 .* σ, μ .+ 2 .* σ
end
