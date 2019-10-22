abstract type MultiOutputMean <: Mean end

"""
    MultiMean <: MultiOutputMean

Mean for multi-output problems. Takes in a vector of `Mean`s, with each representing a
different output.
"""
struct MultiMean <: MultiOutputMean
    m::Vector{Mean}
end
(mm::MultiMean)(x) = hcat([m(x) for m in mm.m]...)
(am::Array{T, 1} where T<:Mean)(x) = hcat([m(x) for m in am]...)
Base.show(io::IO, k::MultiMean) = print(io, "$(k.m)")

"""
    LMMPosMean <: MultiOutputMean

Posterior mean for the Linear Mixing Model.
"""
struct LMMPosMean <: MultiOutputMean
    k::LMMKernel
    x
    Z
    y
    LMMPosMean(k, x, Z, y) = new(k, Fixed(x), Fixed(Z), Fixed(y))
end
Base.show(io::IO, k::LMMPosMean) = print(io, "Posterior($(k.k))")
function (μ::LMMPosMean)(x)
    # NOTE: Our LMM notes have all been derived assuming a different convention for inputs
    # and outputs. Thus, inside this function we will convert them to old conventions and
    # convert back before returning the output.
    yt = unwrap(μ.y)'

    m = unwrap(μ.k.m)
    H = unwrap(μ.k.H)
    p = unwrap(μ.k.p)
    n = size(x, 1)
    n_d = size(unwrap(μ.x), 1)
    σ² = unwrap(μ.k.σ²)
    σ² = isa(σ², Float64) ? ones(p, 1) * σ² : reshape(σ², p, 1)
    Z = unwrap(μ.Z)

    Kx = [k(x, unwrap(μ.x)) for k in μ.k.ks]

    yiσ² = yt ./ σ²
    HiΛy = reshape(H' * yiσ², n_d * m, 1)
    iΛy = reshape(yiσ², n_d * p, 1)
    μ = zeros(n * p, 1)
    z = Z * (Z' * HiΛy)
    for i = 1:m
        μ .+= kron_lmul_lr(Kx[i], reshape(H[:, i], p, 1),
                           kron_lid_lmul(reshape(H[:, i], 1, p), iΛy))
        μ .-= kron_lmul_lr(Kx[i], reshape(H[:, i], p, 1),
                           kron_lid_lmul(reshape(H[:, i], 1, p) * (H ./ σ²), z))
    end
    return reshape(μ, p, n)' # Careful here, inverting p and n is different from transposing
end

"""
    OLMMPosMean <: MultiOutputMean

Posterior mean for the Orthogonal Linear Mixing Model.
"""
struct OLMMPosMean <: MultiOutputMean
    k::OLMMKernel
    ms::Vector{<:Mean}
    x
    y
    OLMMPosMean(k, ms, x, y) = new(k, ms, Fixed(x), Fixed(y))
end
function (m::OLMMPosMean)(x)
    H = unwrap(m.k.H)
    ms = m.ms
    # compute posterior latent means
    μs = [m(x) for m in ms]
    # mix
    return hcat(μs...) * H'
end

function stack(m::Vector)
    s1 = length(m)
    s2 = size(m[1])
    jlim = size(m[1], 2)
    s = (jlim, s1 * s2[1])
    out = hcat(m...)'[:] # Can't use vec here because Nabla
    if s[1] > 1
        out = reshape(out, s...)'
    end
    return out
end

@unionise function unstack(m::Vector{Float64}, p::Int)
    p == 1 && return m
    l = length(m)
    mod(l, p) != 0 && throw(DimensionMismatch("Can't unstack using these dimensions!"))
    return reshape(m, (p, Int(l/p)))'
end

# Matrix-Mean multiplications
Base.:*(m::Matrix, μ::Array{T} where T<:Mean) = MultiMean(m * convert(Array{Any, 1}, μ))
Base.:*(μ::Array{T} where T<:Mean, m::Matrix) = m * μ
Base.:*(m::Array, μ::Mean) = MultiMean(broadcast(*, m, μ))
Base.:*(μ::Mean, m::Array) = m * μ
