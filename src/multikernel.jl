abstract type MultiOutputKernel <: Kernel end
isMulti(k::MultiOutputKernel) = true

function Statistics.var(k::MultiOutputKernel, x)
    return reduce(hcat, [diag(k(x[i, :])) for i in 1:size(x, 1)])'
end

function Statistics.var(k::MultiOutputKernel, x::AbstractDataFrame)
    return reduce(hcat, [diag(k(DataFrame(r))) for r in eachrow(x)])'
end

Base.size(k::MultiOutputKernel, i::Int) = size(k([1.0]), i)

function hourly_cov(k::MultiOutputKernel, x)
    ks = [k(x[i, :]) for i in 1:size(x, 1)]
    return BlockDiagonal(ks)
end

function hourly_cov(k::MultiOutputKernel, x::AbstractMatrix{<:Real})
    ks = [k(x[i, :]') for i in 1:size(x, 1)]
    return BlockDiagonal(ks)
end

function hourly_cov(k::MultiOutputKernel, x::DataFrame)
     ks = [k(DataFrame(r)) for r in eachrow(x)]
    return BlockDiagonal(ks)
end

"""
    NoiseKernel <: MultiOutputKernel

Kernel that treats noisy and noise-free measurements. If both inputs are of the type
`Observed`, covariance will be computed with noise. If any of the inputs is of the type
`Latent`, no noise will be added.

# Constructor:
    NoiseKernel(k_true, k_noise)

- `k_true`: Kernel for noise-free measurements.
- `k_noise`: Kernel to be added to `k_true` in case of noisy observations.
"""
mutable struct NoiseKernel <: MultiOutputKernel
    k_true::Kernel
    k_noise::Kernel
end
Base.show(io::IO, k::NoiseKernel) = print(io, "NoiseKernel($(k.k_true), $(k.k_noise))")
isMulti(k::NoiseKernel) = isMulti(k.k_true)
(k::NoiseKernel)(x::Input, y::Input) = k.k_true(x.val, y.val)
function (k::NoiseKernel)(x::Observed, y::Observed)
    return (k.k_true + k.k_noise)(x.val, y.val)
end
function (k::NoiseKernel)(x::_Observed, y::_Observed)
    return (k.k_true + k.k_noise)(x.val, y.val)
end
function Statistics.var(k::NoiseKernel, x::Input)
    if isa(x.val, DataFrame)
        return reduce(hcat, [diag(k(typeof(x)(xx))) for xx in eachrow(x.val)])'
    else
        # Calling `Matrix` here to avoid the annoying Adjoint type that ruins dispatch.
        eachrow_x = (Matrix(x.val[i, :]') for i in 1:size(x.val, 1))
        return reduce(hcat, [diag(k(typeof(x)(xx))) for xx in eachrow_x])'
    end
end
function hourly_cov(k::NoiseKernel, x::Input)
    isMulti(k) || return k(x)
    ks = [k(typeof(x)(xx)) for xx in x.val]
    return BlockDiagonal(ks)
end

function (k::NoiseKernel)(x, y)
    K = [(k.k_true + k.k_noise) k.k_true; k.k_true k.k_true]
    return MultiKernel(K)(x, y)
end
Statistics.var(k::NoiseKernel, x) = stack([var(k.k_true + k.k_noise, x), var(k.k_true, x)])
# This one is only necessary to break method ambiguity.
function Statistics.var(k::NoiseKernel, x::DataFrame)
    return stack([var(k.k_true + k.k_noise, x), var(k.k_true, x)])
end
function hourly_cov(k::NoiseKernel, x)
    ks = [k(xx) for xx in x]
    return BlockDiagonal(ks)
end

function (k::NoiseKernel)(x::Vector{<:Input}, y::Vector{<:Input}) # Doesn't look efficient, but
    # it also seems inefficient to feed this kind of input.
    lx, ly = length(x), length(y)
    M = Matrix(undef, lx, ly)
    for j in 1:ly
        for i in 1:lx
            M[i, j] = k(x[i], y[j])
        end
    end
    return fuse(M)
end
(k::NoiseKernel)(x::Vector{<:Input}, y::Input) = k(x, [y])
(k::NoiseKernel)(x::Input, y::Vector{<:Input}) = k([x], y)
(k::NoiseKernel)(x) = k(x, x)
function Statistics.var(k::NoiseKernel, x::Vector{Input})
    ks = [k(x[i, :]) for i in 1:size(x, 1)]
    p = Int(size(ks[1], 1) / size(x[1], 1)) # number of outputs
    return vcat([reshape(diag(kk), p, Int(size(kk, 1) / p))' for kk in ks]...)
end
function hourly_cov(k::NoiseKernel, x::Vector{Input})
    ks = [k(xx) for xx in x]
    p = Int(size(ks[1], 1) / size(x[1], 1)) # number of outputs
    l = p * sum([size(xx, 1) for xx in x])
    out = zeros(l, l)
    off = 0
    for i in 1:length(ks)
        s = off + 1
        lx = p * size(x[i], 1)
        out[s:(s + lx - 1), s:(s + lx - 1)] = ks[i]
        off += lx
    end
    return out
end
is_not_noisy(k::NoiseKernel) = false
# TODO: Allow the mixing of typed and untyped `Input`s.
# TODO: Make this function also output BlockDiagonal.

function fuse_equal(m::Matrix{T}) where {T}
    s1 = size(m)
    s2 = size(m[1])
    s = s1 .* s2
    out = Matrix{eltype(T)}(undef, s...)
    for j in 1:s1[2]
        for i in 1:s1[1]
            out[(i-1)*s2[1]+1:i*s2[1], (j-1)*s2[2]+1:j*s2[2]] = m[i, j]
        end
    end
    return out
end

function fuse_equal(m::Matrix{T}) where {T<:Branch}
    # This only supports one element in the matrix at the moment.
    @assert size(m) == (1,1) && return m[1,1]
end

function fuse(m::Matrix)
    sizes = Matrix(undef, size(m)...)
    for j in 1:size(m, 2)
        for i in 1:size(m, 1)
            sizes[i, j] = size(m[i, j])
        end
    end
    all(sizes .== [size(m[1])]) && return fuse_equal(m)
    tsize = (
        sum(Int.([s[1] for s in sizes[:, 1]])), + sum(Int.([s[2] for s in sizes[1, :]]))
    )
    out = Matrix{Float64}(undef, tsize...)
    function offset(i, j)
        oi = sum(Int[s[1] for s in sizes[1:(i-1), j]])
        oj = sum(Int[s[2] for s in sizes[i, 1:(j-1)]])
        return (oi, oj)
    end
    for j in 1:size(m, 2)
        for i in 1:size(m, 1)
            off = offset(i, j)
            out[
                off[1] + 1:off[1] + sizes[i, j][1],
                off[2] + 1:off[2] + sizes[i, j][2]
            ] = m[i, j]
        end
    end
    return out
end

function stack(m::Matrix)
   s1 = size(m)
   s2 = size(m[1])
   s = s1 .* s2
   out = Matrix{Float64}(undef, s...)
   for j in 1:s2[2] # i,j loop over data points
       for i in 1:s2[1]
           for l in 1:s1[2] # k,l loop over base kernels
               for k in 1:s1[1]
                   out[s1[1] * (i - 1) + k, s1[2] * (j - 1) + l] = m[k, l][i, j]
               end
           end
       end
   end
   return out
end
# TODO: Consider using a tensorial representation in order to simply multikernels.


function stack(m::BlockDiagonal)
   s1 = length(blocks(m))
   s2 = size(blocks(m)[1])
   s = s1 .* s2
   out = zeros(s...)
   for j in 1:s2[2] # i,j loop over data points
       for i in 1:s2[1]
           for l in 1:s1
               out[s1 * (i - 1) + l, s1 * (j - 1) + l] = blocks(m)[l][i, j]
           end
       end
   end
   return out
end

"""
    MultiKernel <: MultiOutputKernel

General multi-output kernel.

# Constructor:
    MultiKernel(k::Matrix{Kernel})

Create a kernel that computes the full covariance matrix for each of the kernels, as a block
matrix.
"""
mutable struct MultiKernel <: MultiOutputKernel
    k::Matrix{Kernel}
end
Base.show(io::IO, k::MultiKernel) = print(io, "$(k.k)")
function (m::MultiKernel)(x, y)
    tmp = [k(x, y) for k in m.k]
   return stack(tmp)
end
(m::MultiKernel)(x::Real, y::Real) = m([x], [y])
(m::MultiKernel)(x, y, i::Int, j::Int) = m.k[i, j](x, y)
(m::MultiKernel)(x) = m(x, x)
(m::MultiKernel)(x, i::Int, j::Int) = m(x, x, i, j)
Base.:+(m::MultiKernel, k::Kernel) = MultiKernel(m.k .+ k) ## ref
Base.:+(k::Kernel, m::MultiKernel) = m + k
Base.:+(m1::MultiKernel, m2::MultiKernel) = SumKernel(m1, m2)
Base.size(k::MultiKernel, i::Int) = size(k.k, i)
isMulti(k::MultiKernel) = size(k.k) != (1, 1) || isMulti(k.k[1])

function Base.:+(m::MultiKernel, k::ScaledKernel)
    unwrap(k.scale) ≈ 0.0 && return m
    isMulti(k) && return SumKernel(m, k)
    return MultiKernel(m.k .+ k)
end
Base.:+(k::ScaledKernel, m::MultiKernel) = m + k
Base.:+(m::MultiKernel, k::SumKernel) = isMulti(k) ? SumKernel(m, k) : MultiKernel(m.k .+ k)
Base.:+(k::SumKernel, m::MultiKernel) = m + k

# Leaving this here simply commented out for in case someone thinks it might be worth fixing
# in the future. But this is pretty useless anyway.
# """
#     verynaiveLMMKernel(m, p, σ², H, k)
#
# The most naive way of doing the LMM. Basically generates the full multi-dimensional kernel
# and returns it as a general MultiKernel. Don't expect it to be efficient.
# """
# function verynaiveLMMKernel(m, p, σ², H, k)
#     K = Matrix{Kernel}(undef, m, m)
#     K .= 0
#     for i in 1:m
#         K[i, i] = k
#     end
#     # this here is wrong, because it puts noise even for different timestamps.
#     K = H * K * H' + σ² * Eye(p)
#     return MultiKernel(K)
# end

"""
    NaiveLMMKernel <: MultiOutputKernel

Kernel for the naive implementation of the LMM. Stores a MultiKernel, but only uses the
mixing matrix after the covariance has been computed. Still inefficient, but better.
"""
mutable struct NaiveLMMKernel <: MultiOutputKernel
    H
    σ²
    k::MultiKernel
end
Base.show(io::IO, k::NaiveLMMKernel) = print(io, "$(k.k)")
function NaiveLMMKernel(m, σ², H, k)
    K = Matrix{Kernel}(undef, m, m)
    K .= 0
    for i in 1:m
        K[i, i] = k
    end
    return NaiveLMMKernel(Fixed(H), Positive(σ²), MultiKernel(K))
end
function (k::NaiveLMMKernel)(x, y)
    H = unwrap(k.H)
    σ² = unwrap(k.σ²)
    p = size(H, 1)
    mask = DiagonalKernel()(x, y)
    Λ = fuse(mask .* [Diagonal(fill(σ², p))])
    return kron_lid_lmul(H, kron_lid_lmul(H, k.k(x, y))')' .+ Λ
end
(k::NaiveLMMKernel)(x) = k(x, x)
isMulti(k::NaiveLMMKernel) = isMulti(k.k)

"""
    LMMKernel <: MultiOutputKernel

Kernel that corresponds to the Linear Mixing Model.

* Fields:

- `m`: Number of latent processes
- `p`: Number of outputs
- `σ²`: Variance. Same for all processes if single value, otherwise, vector.
- `H`: Mixing matrix of shape `p`x`m`
- `ks`: Vector containing the kernel for each latent process

* Constructors:

    LMMKernel(m, p, σ², H, k::Vector{Kernel})

Default constructor.

    LMMKernel(m::Int, p::Int, σ²::Float64, H::Matrix, k::Kernel)

Make `m`, `p` and `H` `Fixed` and `σ²` `Positive`, while repeating `k` for all latent
processes.
"""
mutable struct LMMKernel <: MultiOutputKernel
    m # Number of latent processes
    p # Number of outputs
    σ² # Variance. Same for all latent processes if float, otherwise, vector of floats.
    H # Mixing matrix, (p x m)
    ks::Vector{Kernel} # Kernels for the latent processes, m-long

    global function _unsafe_LMMKernel(
        m, # Number of latent processes
        p, # Number of outputs
        σ², # Variance. Same for all latent processes if float, otherwise, vector of floats.
        H, # Mixing matrix, (p x m)
        ks::Vector{<:Kernel}, # Kernels for the latent processes, m-long
    )

        return new(
            m, # Number of latent processes
            p, # Number of outputs
            σ², # Variance. Same for all latent processes if float, otherwise, vector of floats.
            H, # Mixing matrix, (p x m)
            ks, # Kernels for the latent processes, m-long
        )
    end

    function LMMKernel(
        m, # Number of latent processes
        p, # Number of outputs
        σ², # Variance. Same for all latent processes if float, otherwise, vector of floats.
        H, # Mixing matrix, (p x m)
        ks::Vector{<:Kernel}, # Kernels for the latent processes, m-long
    )
        # Bunch of consistency checks
        n_m = unwrap(m)
        n_p = unwrap(p)
        s_H = size(unwrap(H))
        n_k = length(ks)

        s_H == (0, 0) && info(LOGGER, "Initialising LMMKernel with placeholder `H`.")
        n_k == 0 && info(LOGGER, "Initialising LMMKernel with placeholder `ks`.")

        (s_H != (0, 0) && s_H != (n_p, n_m)) && throw(
            ArgumentError("Expected `H` of size ($n_p, $n_m), got $s_H.")
        )
        (n_k != 0 && n_k != n_m) && throw(
            ArgumentError("""
                Expected $n_m kernels, got $n_k. Every latent process should have a kernel.
            """)
        )

        return new(
            m, # Number of latent processes
            p, # Number of outputs
            σ², # Variance. Same for all latent processes if float, otherwise, vector of floats.
            H, # Mixing matrix, (p x m)
            ks, # Kernels for the latent processes, m-long
        )
    end
end
create_instance(T::Type{LMMKernel}, args...) = _unsafe_LMMKernel(args...)
Base.size(k::LMMKernel, i::Int) = i < 1 ? BoundsError() : (i < 3 ? unwrap(k.p) : 1)
function Base.show(io::IO, k::LMMKernel)
    print(io, "$(unwrap(k.H)) * $(k.ks) * $(unwrap(k.H)')")
end
function LMMKernel(m::Int, p::Int, σ²::Union{Float64, Vector{Float64}}, H::Matrix, k::Kernel)
    isa(σ², Vector) && length(σ²) != p &&
        throw(DimensionMismatch("Received $(length(σ²)) nodal variances for $p nodes."))
    return LMMKernel(
        Fixed(m),
        Fixed(p),
        Positive(σ²),
        Fixed(H),
        [k for _ in 1:m]
    )
end
isMulti(k::LMMKernel) = unwrap(k.p) > 1

# This is decidedly not the most optimised implementation of this (we could use stuff from
# optimisedalgebra.jl). However, this is a function that will be very rarely called. The
# important implementation is that of the posterior, which should already be optimised.
function (k::LMMKernel)(x, y)
    H = unwrap(k.H)
    Ks = (kron(k.ks[i](x, y), H[:, i] * H[:, i]') for i in 1:unwrap(k.m))
    mask = DiagonalKernel()(x, y)
    Λ = fuse(mask .* [Diagonal(fill(unwrap(k.σ²), unwrap(k.p)))])
    return sum(Ks) + Λ
end
(k::LMMKernel)(x) = k(x, x)

"""
    LMMPosKernel <: MultiOutputKernel

Posterior kernel for the Linear Mixing Model.
"""
mutable struct LMMPosKernel <: MultiOutputKernel
    k::LMMKernel
    x
    Z
    LMMPosKernel(k, x, Z) = new(k, Fixed(x), Fixed(Z))
end
Base.show(io::IO, k::LMMPosKernel) = print(io, "Posterior($(k.k))")
# There is a good amount of repeated code here. We should see if we can clean some, without
# losing optimisations.
Base.size(k::LMMPosKernel, i::Int) = i < 1 ? BoundsError() : (i < 3 ? unwrap(k.k.p) : 1)
function (k::LMMPosKernel)(x, y)
    # NOTE: Our LMM notes have all been derived assuming a different convention for inputs
    # and outputs. Thus, inside this function we will convert them to old conventions and
    # convert back before returning the output.
    m = unwrap(k.k.m)
    H = unwrap(k.k.H)
    p = unwrap(k.k.p)
    σ² = unwrap(k.k.σ²)
    σ² = isa(σ², Float64) ? ones(p) * σ² : σ²
    Z = unwrap(k.Z)

    K = MultiKernel(Diagonal(k.k.ks))
    Σxy = K(x, y)
    Txy = kron_lid_lmul(H, kron_lid_lmul(H, Σxy')')

    Σx = K(unwrap(k.x), x)
    Tx = kron_lid_lmul(H, Σx)'
    Gx = kron_lid_lmul(H, Tx)

    Σy = K(y, unwrap(k.x))
    Ty = kron_lid_lmul(H, Σy)'
    Gy = kron_lid_lmul(Diagonal(σ².^(-1)) * H, Ty)

    W = kron_lid_lmul(Diagonal(σ².^(-1)) * H, Z)
    V = Gx * W * W' * kron_lid_lmul(H, Ty)

    mask = DiagonalKernel()(x, y)
    Λ = fuse(mask .* [Diagonal(σ²)])

    return Txy .- Gx * Gy .+ V .+ Λ
end
isMulti(k::LMMPosKernel) = isMulti(k.k)
function (k::LMMPosKernel)(x)
    # NOTE: Our LMM notes have all been derived assuming a different convention for inputs
    # and outputs. Thus, inside this function we will convert them to old conventions and
    # convert back before returning the output.
    n = size(x, 1)
    m = unwrap(k.k.m)
    H = unwrap(k.k.H)
    p = unwrap(k.k.p)
    σ² = unwrap(k.k.σ²)
    σ² = isa(σ², Float64) ? ones(p) * σ² : σ²
    Z = unwrap(k.Z)

    ks = [k(x) for k in k.k.ks]
    chols = [cholesky(Hermitian(k) + _EPSILON_^2 * Eye(n)).U for k in ks]
    Uₓₓ = stack(BlockDiagonal(chols))
    Tₓₓ = kron_lid_lmul_lt_m(H, Uₓₓ')

    K = MultiKernel(Diagonal(k.k.ks))
    Σₓ = K(x, unwrap(k.x))
    T_x = kron_lid_lmul(H, Σₓ)'
    G = kron_lid_lmul(Diagonal(σ².^(-1/2)) * H, T_x)

    V = Z' * kron_lid_lmul(H' * Diagonal(σ².^(-1)) * H, T_x)

    Λ = Diagonal(vcat(fill(σ², n)...))

    return Tₓₓ * Tₓₓ' .- G' * G .+ V' * V .+ Λ
end

# TODO: optimise the LMM covariance matrix only for hourly blocks.
Statistics.var(k::LMMPosKernel, x) = _LMMPosvar(k, x)
# The next one is just to break a method ambiguity
Statistics.var(k::LMMPosKernel, x::AbstractDataFrame) = _LMMPosvar(k, x)

function _LMMPosvar(k::LMMPosKernel, x)
    m = unwrap(k.k.m)
    n = size(x, 1)
    H = unwrap(k.k.H)
    p = unwrap(k.k.p)
    σ² = unwrap(k.k.σ²)
    σ² = isa(σ², Float64) ? ones(p, 1) * σ² : reshape(σ², p, 1)
    Z = unwrap(k.Z)

    Kxx = [kk(x, x) + _EPSILON_ .* Eye(n) for kk in k.k.ks]
    Kx = [kk(x, unwrap(k.x)) for kk in k.k.ks]
    outer_HsiΛ = [outer(H[:, i]) ./ sqrt.(σ²)' for i in 1:m]
    inners = Vector{AbstractMatrix}(undef, m)
    for i = 1:m
        inners[i] = kron_lid_lmul(reshape(H[:, i], 1, p) * (H ./ σ²), Z)
    end
    σ²_pred = zeros(n * p, 1)
    for i = 1:m
        σ²_pred .+= kron(diag(Kxx[i]), diag_outer(H[:, i]))
        for j = i:m
            # We are summing terms of the form (i, j) here, but all (i, j) terms are equal
            # to their (j, i) counterparts, so we just run over j >= i and double count all
            # non-diagonal terms.
            scale = j > i ? 2. : 1.
            σ²_pred .-= scale .* diag_outer_kron(Kx[i], outer_HsiΛ[i], Kx[j], outer_HsiΛ[j])
            W = (Kx[i] * (inners[i] * transpose(inners[j]))) * transpose(Kx[j])
            σ²_pred .+= scale .* kron(diag(W), diag_outer(H[:, i], H[:, j]))
        end
    end
    return (reshape(max.(0, σ²_pred), p, n) .+ σ²)'
end
(k::LMMPosKernel)(x, diagonal::Bool) = diagonal ? Diagonal(var(k, x)'[:]) : k(x)

"""
    OLMMKernel <: MultiOutputKernel

Kernel that corresponds to the Orthogonal Linear Mixing Model.

* Fields:

- `m`: Number of latent processes
- `p`: Number of outputs
- `σ²`: Observation noise variance. Same for all processes (so you should normalise the data
    first)
- `D`: Noise variance for the latent processes.
- `H`: Mixing matrix of shape `p`x`m`
- `P`: Projection matrix. `PH=I`.
- `U`: Orthogonal component of the mixing matrix, i.e. its eigenvectors.
- `S_sqrt`: Eigenvalues of the mixing matrix.
- `ks`: Vector containing the kernel for each latent process

* Constructors:

    OLMMKernel(m, p, σ², H, k::Vector{Kernel})

Default constructor.

    OLMMKernel(m::Int, p::Int, σ²::Float64, H::Matrix, k::Kernel)

Make `m`, `p` and `H` `Fixed` and `σ²` `Positive`, while repeating `k` for all latent
processes.
"""
mutable struct OLMMKernel <: MultiOutputKernel
    m # Number of latent processes
    p # Number of outputs
    σ² # Observation noise
    D # latent noise(s)
    H # Mixing matrix, (p x m)
    P::Fixed # Projection matrix, (m x p). Can only be learned through H, since PH=I.
    U::Fixed # Orthogonal component of the mixing matrix, i.e. its eigenvectors.
    # This is already truncated!
    # U is required to be Fixed because, in order to enforce the constraints, it can
    # only be learned through H.
    S_sqrt::Union{<:Positive, <:Fixed} # Eigenvalues of the latent processes.
    # This is already truncated!
    # S_sqrt is restricted to be Positive or Fixed because it can never be non-positive.
    ks::Union{<:Kernel, Vector{<:Kernel}} # Kernels for the latent processes, m-long or the same for all

    global function _unsafe_OLMMKernel(
        m, # Number of latent processes
        p, # Number of outputs
        σ², # Observation noise
        D, # latent noise(s)
        H, # Mixing matrix, (p x m)
        P, # Projection matrix, (m x p)
        U, # Orthogonal component of the mixing matrix. This is already truncated!
        S_sqrt, # Eigenvalues of the latent processes. This is already truncated!
        ks::Union{Kernel, Vector{<:Kernel}}, # Kernels for the latent processes, m-long or the same for all
    )
        return new(
            m, # Number of latent processes
            p, # Number of outputs
            σ², # Observation noise
            D, # latent noise(s)
            H, # Mixing matrix, (p x m)
            P, # Projection matrix, (m x p)
            U, # Orthogonal component of the mixing matrix. This is already truncated!
            S_sqrt, # Eigenvalues of the latent processes. This is already truncated!
            ks, # Kernels for the latent processes, m-long or the same for all
        )
    end

    function OLMMKernel(
        m, # Number of latent processes
        p, # Number of outputs
        σ², # Observation noise
        D, # latent noise(s)
        H, # Mixing matrix, (p x m)
        P, # Projection matrix, (m x p). Can only be learned through H, since PH=I.
        U, # Orthogonal component of the mixing matrix, i.e. its eigenvectors.
        S_sqrt, # Eigenvalues of the latent processes.
        ks::Union{Kernel, Vector{<:Kernel}}, # Kernels for the latent processes, m-long or the same for all
    )
        # Do a bunch of consistency checks
        s_H = size(unwrap(H))
        n_m = unwrap(m)
        n_p = unwrap(p)
        s_P = size(unwrap(P))
        s_U = size(unwrap(U))
        l_S_sqrt = length(unwrap(S_sqrt))
        n_k = isa(ks, Kernel) ? 1 : length(ks)

        s_H == (0, 0) && info(LOGGER, "Initialising OLMMKernel with placeholder `H`.")
        s_P == (0, 0) && info(LOGGER, "Initialising OLMMKernel with placeholder `P`.")
        s_U == (0, 0) && info(LOGGER, "Initialising OLMMKernel with placeholder `U`.")
        !isa(unwrap(S_sqrt), Vector) && throw(
            ArgumentError("`S_sqrt` must be a `Vector` with the norms of the columns of `H`.")
        )
        l_S_sqrt == 0 && info(LOGGER, "Initialising OLMMKernel with placeholder `S_sqrt`.")
        n_k == 0 && info("Initialising OLMMKernel with placeholder `ks`.")

        (n_k != 0 && n_k != 1 && n_k != n_m) && throw(
            ArgumentError("""
                Expected $n_m kernels, got $(n_k). Each latent process needs a kernel.
            """)
        )
        (s_H != (0, 0) && s_H != (n_p, n_m)) && throw(
            ArgumentError("""
                Expected `H` of size ($n_p, $n_m), got $s_H.
            """)
        )
        (s_P != (0, 0) && s_P != (n_m, n_p)) && throw(
            ArgumentError("""
                Expected `P` of size ($n_m, $n_p), got $s_P.
            """)
        )
        (s_U != (0, 0) && s_U != (n_p, n_m)) && throw(
            ArgumentError("""
                Expected `U` of size ($n_p, $n_m), got $s_U.
            """)
        )
        (l_S_sqrt != 0 && l_S_sqrt != n_m) && throw(
            ArgumentError("""
                Expected $n_m eigenvalues, got $(l_S_sqrt). Each latent process needs a value.
            """)
        )
        (n_k > 1 && l_S_sqrt != n_k && !(unwrap(S_sqrt) ≈ ones(l_S_sqrt))) && throw(
            ArgumentError("""
                Unless `S_sqrt` is identically ones, a kernel must be specified for each
                latent process. One can only use the same kernel object for all latent
                processes when `S_sqrt == ones(m)`.
            """)
        )
        !(unwrap(H) ≈ unwrap(U) * Diagonal(unwrap(S_sqrt))) && throw(ArgumentError(
            """
            The mixing matrix, `H`, provided is not of the form `H = U * S`, with `U`
            orthogonal and `S` diagonal. The OLMM requires such form.
            """
        ))

        return new(
            m, # Number of latent processes
            p, # Number of outputs
            σ², # Observation noise
            D, # latent noise(s)
            H, # Mixing matrix, (p x m)
            P, # Projection matrix, (m x p)
            U, # Orthogonal component of the mixing matrix. This is already truncated!
            S_sqrt, # Eigenvalues of the latent processes. This is already truncated!
            ks, # Kernels for the latent processes, m-long or the same for all
        )
    end
end
create_instance(T::Type{OLMMKernel}, args...) = _unsafe_OLMMKernel(args...)
function build_H_and_P(U, S_sqrt)
    H = U * Diagonal(S_sqrt)
    P = Diagonal(S_sqrt.^(-1.0)) * U'
    return H, P
end
function OLMMKernel( # Initialise with H. IT HAS TO BE OF THE FORM `U * S`, with `U`
    # orthogonal and `S` diagonal and positive.
    m::Int,
    p::Int,
    σ²::Float64,
    D::Union{Float64, Vector{Float64}},
    H::AbstractMatrix,
    ks::Union{<:Kernel, Vector{<:Kernel}}
)
    S_sqrt = sqrt.(diag(H' * H))
    U = H * Diagonal(S_sqrt .^ (-1.0))
    _, P = build_H_and_P(U, S_sqrt)
    return OLMMKernel(
        Fixed(m),
        Fixed(p),
        Positive(σ²),
        Positive(D),
        H,
        Fixed(P),
        Fixed(U),
        Positive(S_sqrt),
        ks
    )
end
function OLMMKernel( # Initialise with U and S
    m::Int,
    p::Int,
    σ²::Float64,
    D::Union{Float64, Vector{Float64}},
    U::AbstractMatrix,
    S_sqrt::Vector{Float64},
    ks::Union{<:Kernel, Vector{<:Kernel}}
)
    (size(U, 2) != size(S_sqrt, 1) || size(S_sqrt, 1) != m) &&
        notice(LOGGER, "U and S_sqrt must be truncated to m.")
    H, P = build_H_and_P(U, S_sqrt)
    return OLMMKernel(
        Fixed(m),
        Fixed(p),
        Positive(σ²),
        Positive(D),
        H,
        Fixed(P),
        Fixed(U),
        Positive(S_sqrt),
        ks
    )
end
"""
    greedy_U(k::OLMMKernel, x, y)

Compute the greedy solution for the optimal `U` matrix of the OLMM model.
"""
function greedy_U(k::OLMMKernel, x, y)
    n = size(x, 1)
    σ² = unwrap(k.σ²)
    m = unwrap(k.m)
    S_sqrt = unwrap(k.S_sqrt)
    D = unwrap(k.D)
    D = isa(D, Vector) ? D : ones(m) .* D

    function Σ(i)
        K = S_sqrt[i] * k.ks[i](x) + σ² * I + S_sqrt[i] * D[i] * I
        Z = cholesky(K).L \ y
        return (y' * y) ./ σ² - Z' * Z
    end

    U, _, _ = svd(Σ(1))
    us = [U[:, 1]]
    V = U[:, 2:end]

    for i in 2:m
        U, _, _ = svd(V' * Σ(i) * V)
        push!(us, V * U[:, 1])
        V = V * U[:, 2:end]
    end

    return hcat(us...)
end
@unionise function optk(k, x) # This is an optimised implementation of the OLMM for when all kernels
# are the same
    # compute latent covariances
    K = k.ks(x)
    # mix them
    n = size(x, 1)
    H = unwrap(k.H)
    σ² = unwrap(k.σ²)
    D = unwrap(k.D)
    m = unwrap(k.m)
    D = isa(D, Float64) ? fill(D, m) : D
    out = Matrix(undef, n, n)
    for j in 1:n
        for i in j:n
            out[i, j] =  H * Diagonal(fill(K[i, j], m)) * H'
            out[j, i] = out[i, j]
        end
    end
    # add noises
    for j in 1:n
        out[j, j] += σ² * I + H * Diagonal(D) * H'
    end
    # build big mixed matrix
    return fuse_equal(out)
end

function (k::OLMMKernel)(x, y)
    # Use more specific implementation if possible.
    x === y && return k(x)
    # isa(k.ks, Kernel) && return optk(k, x, y) TODO: implement this (not necessary for now)
    # compute latent proc covs
    n1 = size(x, 1)
    n2 = size(y, 1)
    H = unwrap(k.H)
    σ² = unwrap(k.σ²)
    p = unwrap(k.p)
    D = unwrap(k.D)
    m = unwrap(k.m)
    D = isa(D, Float64) ? fill(D, m) : D
    Σs = [lk(x, y) for lk in k.ks]
    mix_Σs = Matrix(undef, n1, n2)
    for j in 1:n2
        for i in 1:n1
            mix_Σs[i, j] = H * Diagonal([s[i, j] for s in Σs]) * H'
        end
    end
    # add noises
    noise_mask = DiagonalKernel()(x, y)
    noise = σ² * Eye(p) + H * Diagonal(D) * H'
    # This is ugly, but we do not want a Hadamard product. Instead, we want every non-zero
    # entry of `noise_mask` to be equal to the entire matrix `noise`.
    mix_Σs += noise_mask .* [noise]
    # build big mixed matrix
    return fuse_equal(mix_Σs)
end
function (k::OLMMKernel)(x)
    isa(k.ks, Kernel) && return optk(k, x)
    # compute latent proc covs
    n = size(x, 1)
    H = unwrap(k.H)
    σ² = unwrap(k.σ²)
    p = unwrap(k.p)
    D = unwrap(k.D)
    m = unwrap(k.m)
    D = isa(D, Float64) ? fill(D, m) : D
    Σs = [lk(x) for lk in k.ks]
    # mix them
    T = eltype(Σs)
    mix_Σs = Matrix{T}(undef, n, n)
    for j in 1:n
        for i in j:n
            v = [s[i:i, j] for s in Σs]
            # This vcat(...) looks redundant, but is essential for Nabla
            mix_Σs[i, j] = H * Diagonal(vcat(v...)) * H'
            mix_Σs[j, i] = mix_Σs[i, j]
        end
    end
    # add noises
    for j in 1:n
        mix_Σs[j, j] += σ² * Eye(p) + H * Diagonal(D) * H'
    end
    # build big mixed matrix
    return fuse_equal(mix_Σs)
end
isMulti(k::OLMMKernel) = unwrap(k.p) > 1
Base.size(k::OLMMKernel, i::Int) = i < 1 ? BoundsError() : (i < 3 ? unwrap(k.p) : 1)

"""
    LSOLMMKernel <: MultiOutputKernel

Latent Space OLMM kernel, which uses positions defined in a latent space to define the
mixing matrix `H`. These latent positions are input to a kernel, `Hk`, which outputs a
distance matrix whose `m` eigenvectors and eigenvalues construct `H`. This reduces the
number of free parameters in the mixing matrix and mitigates overfitting.

* Fields:

- `Hk`: Kernel used to build the mixing matrix.
- `lat_pos`: Latent positions corresponding to each latent process.
- `out_pos`: Output positions corresponding to each output process.
- `olmm`: Corresponding `OLMMKernel`.

The mixing matrix `H` is defined as

    M = Hk(out_pos, lat_pos)
    U, S, _ = svd(M)
    S_sqrt = sqrt.(S)
    H = U * Diagonal(S_sqrt)

Any attribute of the `OLMMKernel` can be directly accessed from the `LSOLMMKernel`, i.e.

    julia> k = LSOLMMKernel(
                    Fixed(3), Fixed(5), Fixed(0.02), Fixed([0.05 for i in 1:3]),
                    stretch(EQ(), Positive(5.0)), rand(3), rand(5),
                    [stretch(EQ(), Positive(5.0)) for i in 1:3]
                );

    julia> k.H ≈ k.olmm.H
    true

    julia> k.D ≈ k.olmm.D
    true

    julia> k.m ≈ k.olmm.m
    true

* Constructors:

    LSOLMMKernel(Hk::Kernel, lat_pos, out_pos, olmm::OLMMKernel)

Default constructor.

    LSOLMMKernel(m, p, σ², D, Hk::Kernel, lat_pos, out_pos, ks::Union{Kernel, Vector{<:Kernel}})

Build an `LSOLMMKernel` on top of a `OLMMKernel` that is defined by `m`, `p`, `σ²`, D, and
`ks` and by the mixing matrix defined by `Hk`, `out_pos`, and `lat_pos`.
"""
mutable struct LSOLMMKernel <: MultiOutputKernel
    Hk::Kernel # Kernel for building mixing matrix
    lat_pos # Latent positions for latent processes
    out_pos # Latent positions for output processes
    olmm::OLMMKernel # Corresponding OLMM kernel

    global function _unsafe_LSOLMMKernel(
        Hk::Kernel, # Kernel for building mixing matrix
        lat_pos, # Latent positions for latent processes
        out_pos, # Latent positions for output processes
        olmm::OLMMKernel,
    )
        return new(
            Hk, # Kernel for building mixing matrix
            lat_pos, # Latent positions for latent processes
            out_pos, # Latent positions for output processes
            olmm,
        )
    end

    function LSOLMMKernel(
        Hk::Kernel, # Kernel for building mixing matrix
        lat_pos, # Latent positions for latent processes
        out_pos, # Latent positions for output processes
        olmm::OLMMKernel,
    )
        k = new(
            Hk, # Kernel for building mixing matrix
            lat_pos, # Latent positions for latent processes
            out_pos, # Latent positions for output processes
            olmm,
        )
        update_LSOLMM!(k)
        return k
    end

    function LSOLMMKernel(
        m, # Number of latent processes
        p, # Number of outputs
        σ², # Observation noise
        D, # latent noise(s)
        Hk::Kernel, # Kernel for building mixing matrix
        lat_pos, # Latent positions for latent processes
        out_pos, # Latent positions for output processes
        ks::Union{Kernel, Vector{<:Kernel}}, # Kernels for the latent processes, m-long or the same for all
        S_sqrt=Fixed(ones(unwrap(m))), # By default incorporate in the latent process variances
    )
        # Do a bunch of consistency checks
        n_m = unwrap(m)
        n_p = unwrap(p)
        n_k = isa(ks, Kernel) ? 1 : length(ks)
        n_out = size(unwrap(out_pos), 1)
        n_lat = size(unwrap(lat_pos), 1)

        # We must be able to initialise the kernel without knowing the number of outputs
        # because that's what happens in prod.
        if n_out == 0
            info(
                LOGGER,
                "Initialising OLMMKernel with placeholder `H`, `P`, `U`, and `S_sqrt`."
            )
        elseif n_out != n_p
            throw(
                ArgumentError(
                    "Expected $n_p latent positions for the outputs. Got $n_out."
                )
            )
        end

        if n_lat != n_m
            throw(
                ArgumentError(
                    "Expected $n_m latent positions for the latent processes. Got $n_lat."
                )
            )
        end

        n_k == 0 && info("Initialising OLMMKernel with placeholder `ks`.")

        (n_k != 0 && n_k != 1 && n_k != n_m) && throw(
            ArgumentError("""
                Expected $n_m kernels, got $(n_k). Each latent process needs a kernel.
            """)
        )

        H0, P0 = if n_out == 0
            [Fixed(Matrix{Float64}(undef, 0, 0)) for i in 1:2] # Placeholders
        else
            Fixed.(_build_H_from_kernel(Hk, unwrap(out_pos), unwrap(lat_pos)))
        end

        U = if n_out == 0
            Fixed(Matrix{Float64}(undef, 0, 0)) # Placeholder
        else
           H0
        end

        H, P = if size(H0, 1) > 0
            Fixed.([
                unwrap(H0) * Diagonal(unwrap(S_sqrt)),
                Diagonal(unwrap(S_sqrt).^(-1.0)) * unwrap(P0)
            ])
        else
            H0, P0
        end
# @show H0;@show P0; @show H; @show P
        return new(
            Hk, # Kernel for building mixing matrix
            lat_pos, # Latent positions for latent processes
            out_pos, # Latent positions for output processes
            _unsafe_OLMMKernel(
                m, # Number of latent processes
                p, # Number of outputs
                σ², # Observation noise
                D, # latent noise(s)
                H, # Mixing matrix, (p x m)
                P, # Projection matrix, (m x p)
                U, # Orthogonal component of the mixing matrix. This is already truncated!
                S_sqrt, # Eigenvalues of the latent processes. This is already truncated!
                ks, # Kernels for the latent processes, m-long or the same for all
            )
        )
    end
end
create_instance(T::Type{LSOLMMKernel}, args...) = _unsafe_LSOLMMKernel(args...)

function Base.getproperty(k::LSOLMMKernel, p::Symbol)
    if p in fieldnames(LSOLMMKernel)
        return getfield(k, p)
    else
        return getproperty(k.olmm, p)
    end
end

"""
    _build_H_from_kernel(Hk, out_pos, lat_pos)

Construct the mixing matrix from the kernel `Hk` and the latent positions for the outputs,
`out_pos`, and for the latent processes, `lat_pos`. Here `S_sqrt` is taken as identically
ones, such that it can be optmised independently (or simply incorporated into the variance
of the latent processes). We parameterise the orthogonal matrix `H` as `U * V` (instead of
simply as `U`) to avoid 180 degree rotations of the eigenvectors, making it more numerically
stable.
"""
function _build_H_from_kernel(Hk, out_pos, lat_pos)
    m = length(lat_pos)
    M = Hk(out_pos, lat_pos)
    U, _, V = svd(M)
    H = U * V
    P = H'
    return H, P
end

"""
    update_LSOLMM!(k::LSOLMMKernel)

Use current latent positions stored in `k` to update the mixing matrix. Operates inplace.
"""
function update_LSOLMM!(k::LSOLMMKernel)
    H, P = _build_H_from_kernel(k.Hk, unwrap(k.out_pos), unwrap(k.lat_pos))
    S_sqrt = unwrap(k.S_sqrt)
    k.olmm.U = Fixed(H) # Because S_sqrt
    k.olmm.H = Fixed(H * Diagonal(S_sqrt))
    k.olmm.P = Fixed(Diagonal(S_sqrt.^(-1.0)) * P)
end

(k::LSOLMMKernel)(x, y) = k.olmm(x, y)
(k::LSOLMMKernel)(x) = k.olmm(x)

isMulti(k::LSOLMMKernel) = unwrap(k.p) > 1
Base.size(k::LSOLMMKernel, i::Int) = i < 1 ? BoundsError() : (i < 3 ? unwrap(k.p) : 1)


# Groupped OLMM kernel
"""
    GOLMMKernel <: MultiOutputKernel

Alternative kernel for the Orthogonal Linear Mixing Model, based on groupings
computed using output embeddings.

# Fields
- `group_kernel::Kernel`: Kernel that correlates outputs based on `group_embeddings`.
- `group_embeddings::Vector{<:Real}`: Group embeddings used by the `group_kernel`.
- `olmm_kernel::OLMMKernel`: OLMM kernel computed using the above.

# Constructors
    GOLMMKernel(m, p, σ², d, k::Vector{<:Kernel}, group_kernel::Kernel, group_embeddings)
    GOLMMKernel(group_kernel::Kernel, group_embs, olmm_kernel::OLMMKernel)
"""
mutable struct GOLMMKernel <: MultiOutputKernel
    group_kernel::Kernel
    group_embeddings::Union{Vector{<:Real}, Branch{<:Vector{<:Real}}}
    olmm_kernel::OLMMKernel

    function GOLMMKernel(
        m, # number of latent independent GPs
        p, # number of outputs
        σ², # observation noise
        d, # latent noise
        ks, # kernels for latent independent GPs
        group_kernel, # kernel that correlates outputs based on `group_embs_init`
        group_embeddings, # values of group embbeddings
    )
        olmm_kernel = olmm_kernel_using_groups(
            m,
            p,
            σ²,
            d,
            ks,
            group_kernel,
            group_embeddings,
        )

        return new(group_kernel, group_embeddings, olmm_kernel)
    end


    function GOLMMKernel(
        group_kernel, # kernel that correlates outputs based on `ge`
        group_embeddings, # group embeddings
        olmm_kernel, # current OLMM kernel
    )
        # take all parameters, except H, from current OLMM kernel
        p = unwrap(olmm_kernel.p)
        m = unwrap(olmm_kernel.m)
        σ² = olmm_kernel.σ²
        D = olmm_kernel.D
        ks = olmm_kernel.ks

        # make a new OLMM kernel, with H computed using group embeddings
        olmm_kernel_new = olmm_kernel_using_groups(
            m,
            p,
            σ²,
            D,
            ks,
            group_kernel,
            group_embeddings,
        )

        return new(group_kernel, group_embeddings, olmm_kernel_new)
    end

end


"""
    olmm_kernel_using_groups(m, p, σ², d, ks, group_kernel, group_embeddings)

Return an [`OLMMKernel`](@ref), with H computed using group embeddings and group kernel.
"""
function olmm_kernel_using_groups(m, p, σ², d, ks, group_kernel, group_embeddings)

    # construct a p x p Gram matrix using group kernel on group embeddings
    C = group_kernel(group_embeddings) + _EPSILON_ * Eye(p);

    # perform its SVD to compute U and S (analogous to PCA)
    U, S, _ = svd(C)
    U_ = U[:, 1:m]
    S_ = sqrt.(S)[1:m]

    # use the U and S to compute H
    H, P = GPForecasting.build_H_and_P(U_, S_)

    return _unsafe_OLMMKernel(
        Fixed(m),
        Fixed(p),
        σ²,
        d,
        Fixed(H),
        Fixed(P),
        Fixed(U_),
        Fixed(S_),
        ks,
    );
end

isMulti(k::GOLMMKernel) = isMulti(k.olmm_kernel)

(k::GOLMMKernel)(x, y) = k.olmm_kernel(x, y)
(k::GOLMMKernel)(x) = k(x, x)

# Matrix-Kernel multiplications
Base.:*(m::Matrix, k::Kernel) = MultiKernel(m .* k)
Base.:*(k::Kernel, m::Matrix) = m * k

Base.:*(m::Matrix, k::ZeroKernel) = zeros(Kernel, size(m))
Base.:*(k::ZeroKernel, m::Matrix) = m * k

Base.:*(m::Matrix, k::ScaledKernel) = ScaledKernel(k.scale, m * k.k)
Base.:*(k::ScaledKernel, m::Matrix) = ScaledKernel(k.scale, k.k * m)

Base.:*(m::Matrix, k::StretchedKernel) = StretchedKernel(k.stretch, m * k.k)
Base.:*(k::StretchedKernel, m::Matrix) = StretchedKernel(k.stretch, k.k * m)

Base.:*(m::Matrix, k::SumKernel) = SumKernel(m * k.k1, m * k.k2)
Base.:*(k::SumKernel, m::Matrix) = SumKernel(k.k1 * m, k.k2 * m)

Base.:*(m::Matrix, k::PeriodicKernel) = PeriodicKernel(k.T, m * k.k)
Base.:*(k::PeriodicKernel, m::Matrix) = PeriodicKernel(k.T, k.k * m)

Base.:*(m::Matrix, k::MultiKernel) = MultiKernel(m * k.k)
Base.:*(k::MultiKernel, m::Matrix) = MultiKernel(k.k * m)
