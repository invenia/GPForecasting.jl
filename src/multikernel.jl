export NoiseKernel, MultiKernel, LMMKernel, verynaiveLMMKernel, LMMPosKernel,
NaiveLMMKernel, MultiOutputKernel, OLMMKernel

abstract type MultiOutputKernel <: Kernel end
isMulti(k::MultiOutputKernel) = true

var(k::MultiOutputKernel, x) = hcat([diag(k(x[i, :])) for i in 1:size(x, 1)]...)'
diag(x::Real) = x

size(k::MultiOutputKernel, i::Int) = size(k([1.0]), i)

function hourly_cov(k::MultiOutputKernel, x)
     ks = [k(x[i, :]) for i in 1:size(x, 1)]
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
show(io::IO, k::NoiseKernel) = print(io, "NoiseKernel($(k.k_true), $(k.k_noise))")
isMulti(k::NoiseKernel) = isMulti(k.k_true)
(k::NoiseKernel)(x::Input, y::Input) = k.k_true(x.val, y.val)
function (k::NoiseKernel)(x::Observed, y::Observed)
    return (k.k_true + k.k_noise)(x.val, y.val)
end
var(k::NoiseKernel, x::Input) = hcat([diag(k(typeof(x)(xx))) for xx in x.val]...)'
function hourly_cov(k::NoiseKernel, x::Input)
    isMulti(k) || return k(x)
    ks = [k(typeof(x)(xx)) for xx in x.val]
    return BlockDiagonal(ks)
end

function (k::NoiseKernel)(x, y)
    K = [(k.k_true + k.k_noise) k.k_true; k.k_true k.k_true]
    return MultiKernel(K)(x, y)
end
var(k::NoiseKernel, x) = stack([var(k.k_true + k.k_noise, x), var(k.k_true, x)])
function hourly_cov(k::NoiseKernel, x)
    ks = [k(xx) for xx in x]
    return BlockDiagonal(ks)
end

function (k::NoiseKernel)(x::Vector{<:Input}, y::Vector{<:Input}) # Doesn't look efficient, but
    # it also seems inefficient to feed this kind of input.
    lx, ly = length(x), length(y)
    M = Matrix(lx, ly)
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
function var(k::NoiseKernel, x::Vector{Input})
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
# TODO: Allow the mixing of typed and untyped `Input`s.
# TODO: Make this function also output BlockDiagonal.

function fuse_equal(m::Matrix)
   s1 = size(m)
   s2 = size(m[1])
   s = s1 .* s2
   out = Matrix{Float64}(s...)
   for j in 1:s1[2]
       for i in 1:s1[1]
           out[(i-1)*s2[1]+1:i*s2[1], (j-1)*s2[2]+1:j*s2[2]] = m[i, j]
       end
   end
   return out
end

function fuse(m::Matrix)
    sizes = Matrix(size(m)...)
    for j in 1:size(m, 2)
        for i in 1:size(m, 1)
            sizes[i, j] = size(m[i, j])
        end
    end
    all(sizes .== [size(m[1])]) && return fuse_equal(m)
    tsize = (
        sum(Int.([s[1] for s in sizes[:, 1]])), + sum(Int.([s[2] for s in sizes[1, :]]))
    )
    out = Matrix{Float64}(tsize...)
    function offset(i, j)
        oi = sum(Int.([s[1] for s in sizes[1:(i-1), j]]))
        oj = sum(Int.([s[2] for s in sizes[i, 1:(j-1)]]))
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
   out = Matrix{Float64}(s...)
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
show(io::IO, k::MultiKernel) = print(io, "$(k.k)")
function (m::MultiKernel)(x, y)
    tmp = [k(x, y) for k in m.k]
   return stack(tmp)
end
(m::MultiKernel)(x::Real, y::Real) = m([x], [y])
(m::MultiKernel)(x, y, i::Int, j::Int) = m.k[i, j](x, y)
(m::MultiKernel)(x) = m(x, x)
(m::MultiKernel)(x, i::Int, j::Int) = m(x, x, i, j)
(+)(m::MultiKernel, k::Kernel) = MultiKernel(m.k .+ k)
(+)(k::Kernel, m::MultiKernel) = m + k
(+)(m1::MultiKernel, m2::MultiKernel) = SumKernel(m1, m2)
size(k::MultiKernel, i::Int) = size(k.k, i)
isMulti(k::MultiKernel) = size(k.k) != (1, 1) || isMulti(k.k[1])

function (+)(m::MultiKernel, k::ScaledKernel)
    unwrap(k.scale) ≈ 0.0 && return m
    isMulti(k) && return SumKernel(m, k)
    return MultiKernel(m.k .+ k)
end
(+)(k::ScaledKernel, m::MultiKernel) = m + k
(+)(m::MultiKernel, k::SumKernel) = isMulti(k) ? SumKernel(m, k) : MultiKernel(m.k .+ k)
(+)(k::SumKernel, m::MultiKernel) = m + k

"""
    verynaiveLMMKernel(m, p, σ², H, k)

The most naive way of doing the LMM. Basically generates the full multi-dimensional kernel
and returns it as a general MultiKernel. Don't expect it to be efficient.
"""
function verynaiveLMMKernel(m, p, σ², H, k)
    K = Matrix{Kernel}(m, m)
    K .= 0
    for i in 1:m
        K[i, i] = k
    end
    K = H * K * H' + σ² * eye(p)
    return MultiKernel(K)
end

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
show(io::IO, k::NaiveLMMKernel) = print(io, "$(k.k)")
function NaiveLMMKernel(m, σ², H, k)
    K = Matrix{Kernel}(m, m)
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
    n1 = size(x, 1)
    n2 = size(y, 1)
    Λ = fuse(fill(σ² * eye(p), (n1, n2)))
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

        s_H == (0, 0) && warn("Initialising LMMKernel with placeholder `H`.")
        n_k == 0 && warn("Initialising LMMKernel with placeholder `ks`.")

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
size(k::LMMKernel, i::Int) = i < 1 ? BoundsError() : (i < 3 ? unwrap(k.p) : 1)
function show(io::IO, k::LMMKernel)
    print(io, "$(unwrap(k.H)) * $(k.ks) * $(ctranspose(unwrap(k.H)))")
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
show(io::IO, k::LMMPosKernel) = print(io, "Posterior($(k.k))")
# There is a good amount of repeated code here. We should see if we can clean some, without
# losing optimisations.
size(k::LMMPosKernel, i::Int) = i < 1 ? BoundsError() : (i < 3 ? unwrap(k.k.p) : 1)
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

    K = MultiKernel(diagm(k.k.ks))
    Σxy = K(x, y)
    Txy = kron_lid_lmul(H, kron_lid_lmul(H, Σxy')')

    Σx = K(unwrap(k.x), x)
    Tx = kron_lid_lmul(H, Σx)'
    Gx = kron_lid_lmul(H, Tx)

    Σy = K(y, unwrap(k.x))
    Ty = kron_lid_lmul(H, Σy)'
    Gy = kron_lid_lmul(diagm(σ².^(-1)) * H, Ty)

    W = kron_lid_lmul(diagm(σ².^(-1)) * H, Z)
    V = Gx * W * W' * kron_lid_lmul(H, Ty)

    mask = DiagonalKernel()(x, y)
    Λ = fuse(mask .* [diagm(σ²)])

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
    chols = chol.([(Hermitian(k) + _EPSILON_^2 * eye(n)) for k in ks])
    Uₓₓ = stack(BlockDiagonal(chols))
    Tₓₓ = kron_lid_lmul_lt_m(H, Uₓₓ')

    K = MultiKernel(diagm(k.k.ks))
    Σₓ = K(x, unwrap(k.x))
    T_x = kron_lid_lmul(H, Σₓ)'
    G = kron_lid_lmul(diagm(σ².^(-1/2)) * H, T_x)

    V = Z' * kron_lid_lmul(H' * diagm(σ².^(-1)) * H, T_x)

    Λ = diagm(vcat(fill(σ², n)...))

    return Tₓₓ * Tₓₓ' .- G' * G .+ V' * V .+ Λ
end

# TODO: optimise the LMM covariance matrix only for hourly blocks.
function var(k::LMMPosKernel, x)
    m = unwrap(k.k.m)
    n = size(x, 1)
    H = unwrap(k.k.H)
    p = unwrap(k.k.p)
    σ² = unwrap(k.k.σ²)
    σ² = isa(σ², Float64) ? ones(p, 1) * σ² : reshape(σ², p, 1)
    Z = unwrap(k.Z)

    Kxx = [kk(x, x) + _EPSILON_ .* eye(n) for kk in k.k.ks]
    Kx = [kk(x, unwrap(k.x)) for kk in k.k.ks]
    outer_HsiΛ = [outer(H[:, i]) ./ sqrt.(σ²)' for i in 1:m]
    inners = Vector{AbstractMatrix}(m)
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
            W = A_mul_Bt(Kx[i] * A_mul_Bt(inners[i], inners[j]), Kx[j])
            σ²_pred .+= scale .* kron(diag(W), diag_outer(H[:, i], H[:, j]))
        end
    end
    return (reshape(max.(0, σ²_pred), p, n) .+ σ²)'
end
(k::LMMPosKernel)(x, diagonal::Bool) = diagonal ? diagm(var(k, x)'[:]) : k(x)

"""
    OLMMKernel <: MultiOutputKernel

Kernel that corresponds to the Orthogonal Linear Mixing Model.

* Fields:

- `m`: Number of latent processes
- `p`: Number of outputs
- `σ²`: Variance. Same for all processes (so you should normalise the data first)
- `H`: Mixing matrix of shape `p`x`m`
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
    P # Projection matrix, (m x p)
    U # Orthogonal component of the mixing matrix. This is already truncated!
    S_sqrt # Eigenvalues of the latent processes. This is already truncated!
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
        P, # Projection matrix, (m x p)
        U, # Orthogonal component of the mixing matrix. This is already truncated!
        S_sqrt, # Eigenvalues of the latent processes. This is already truncated!
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

        s_H == (0, 0) && warn("Initialising OLMMKernel with placeholder `H`.")
        s_P == (0, 0) && warn("Initialising OLMMKernel with placeholder `P`.")
        s_U == (0, 0) && warn("Initialising OLMMKernel with placeholder `U`.")
        !isa(unwrap(S_sqrt), Vector) && throw(
            ArgumentError("`S_sqrt` must be a `Vector` with the norms of the columns of `H`.")
        )
        l_S_sqrt == 0 && warn("Initialising OLMMKernel with placeholder `S_sqrt`.")
        n_k == 0 && warn("Initialising OLMMKernel with placeholder `ks`.")

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
        !(unwrap(H) ≈ unwrap(U) * diagm(unwrap(S_sqrt))) && throw(ArgumentError(
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
    H = U * diagm(S_sqrt)
    P = diagm(S_sqrt.^(-1.0)) * U'
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
    U = H * diagm(S_sqrt.^(-1.0))
    _, P = build_H_and_P(U, S_sqrt)
    return OLMMKernel(
        Fixed(m),
        Fixed(p),
        Positive(σ²),
        Positive(D),
        Fixed(H),
        Fixed(P),
        Fixed(U),
        Fixed(S_sqrt),
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
        warn("U and S_sqrt must be truncated to m.")
    H, P = build_H_and_P(U, S_sqrt)
    return OLMMKernel(
        Fixed(m),
        Fixed(p),
        Positive(σ²),
        Positive(D),
        Fixed(H),
        Fixed(P),
        Fixed(U),
        Fixed(S_sqrt),
        ks
    )
end
"""
    greedy_U(k::OLMMKernel, x, y)

Compute the greedy solution for the optimal `U` matrix of the OLMM model. This assumes that
`D` (the latent noises) is uniformly diagonal, which is not necessarily true.
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
        Uk = Nabla.chol(K)
        Z = Uk' \ y
        return (y' * y) ./ σ² - Z' * Z
    end

    U, _, _ = svd(Σ(1))
    us = [U[:, end]]
    V = U[:, 1:end - 1]

    for i in 2:m
        U, _, _ = svd(V' * Σ(i) * V)
        push!(us, V * U[:, end])
        V = V * U[:, 1:end - 1]
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
    out = Matrix(n, n)
    for j in 1:n
        for i in j:n
            out[i, j] =  H * diagm(fill(K[i, j], m)) * H'
            out[j, i] = out[i, j]
        end
    end
    # add noises
    for j in 1:n
        out[j, j] += σ² * I + H * diagm(D) * H'
    end
    # build big mixed matrix
    return fuse_equal(out)
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
    mix_Σs = Matrix(n, n)
    for j in 1:n
        for i in j:n
            mix_Σs[i, j] = H * diagm([s[i, j] for s in Σs]) * H'
            mix_Σs[j, i] = mix_Σs[i, j]
        end
    end
    # add noises
    for j in 1:n
        mix_Σs[j, j] += σ² * eye(p) + H * diagm(D) * H'
    end
    # build big mixed matrix
    return fuse_equal(mix_Σs)
end
isMulti(k::OLMMKernel) = unwrap(k.p) > 1
size(k::OLMMKernel, i::Int) = i < 1 ? BoundsError() : (i < 3 ? unwrap(k.p) : 1)

# Matrix-Kernel multiplications
(*)(m::Matrix, k::Kernel) = MultiKernel(m .* k)
(*)(k::Kernel, m::Matrix) = m * k

(*)(m::Matrix, k::ZeroKernel) = zeros(Kernel, size(m))
(*)(k::ZeroKernel, m::Matrix) = m * k

(*)(m::Matrix, k::ScaledKernel) = ScaledKernel(k.scale, m * k.k)
(*)(k::ScaledKernel, m::Matrix) = ScaledKernel(k.scale, k.k * m)

(*)(m::Matrix, k::StretchedKernel) = StretchedKernel(k.stretch, m * k.k)
(*)(k::StretchedKernel, m::Matrix) = StretchedKernel(k.stretch, k.k * m)

(*)(m::Matrix, k::SumKernel) = SumKernel(m * k.k1, m * k.k2)
(*)(k::SumKernel, m::Matrix) = SumKernel(k.k1 * m, k.k2 * m)

(*)(m::Matrix, k::PeriodicKernel) = PeriodicKernel(k.T, m * k.k)
(*)(k::PeriodicKernel, m::Matrix) = PeriodicKernel(k.T, k.k * m)

(*)(m::Matrix, k::MultiKernel) = MultiKernel(m * k.k)
(*)(k::MultiKernel, m::Matrix) = MultiKernel(k.k * m)
