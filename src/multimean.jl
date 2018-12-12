export MultiMean, LMMPosMean, MultiOutputMean, OLMMPosMean

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
show(io::IO, k::MultiMean) = print(io, "$(k.m)")

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
show(io::IO, k::LMMPosMean) = print(io, "Posterior($(k.k))")
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
    ms::Vector{PosteriorMean}
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

function stack(m::Vector) # This ugly, should be replaced soon
   s1 = length(m)
   s2 = size(m[1])
   jlim = isa(m[1], Vector) ? 1 : s2[2]
   s = (s1 * s2[1], jlim)
   out = Array{Float64}(undef, s...)
   for j in 1:jlim # i,j loop over data points
       for i in 1:s2[1]
           for k in 1:s1
               out[s1[1] * (i - 1) + k, (j - 1) + 1] = m[k][i, j]
           end
       end
   end
   return size(out, 2) == 1 ? reshape(out, length(out)) : out
end

function unstack(m::Vector{Float64}, p::Int)
    p == 1 && return m
    l = length(m)
    mod(l, p) != 0 && throw(DimensionMismatch("Can't unstack using these dimensions!"))
    return reshape(m, (p, Int(l/p)))'
end

# Matrix-Mean multiplications
(*)(m::Matrix, μ::Array{T} where T<:Mean) = MultiMean(m * convert(Array{Any, 1}, μ))
(*)(μ::Array{T} where T<:Mean, m::Matrix) = m * μ
(*)(m::Array, μ::Mean) = MultiMean(broadcast(*, m, μ))
(*)(μ::Mean, m::Array) = m * μ
