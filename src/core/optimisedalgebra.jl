module OptimisedAlgebra

using Nabla
import Nabla: chol

export outer, kron_lid_lmul, kron_lid_lmul_lt_s, kron_lid_lmul_lt_m,
kron_rid_lmul_s, kron_rid_lmul_m, kron_lmul_lr, kron_lmul_rl, diag_outer_kron,
diag_outer, Js, Ms, sum_kron_J_ut, sum_kron_M, eye_sum_kron_M_ut, BlockDiagonal, blocks

import Base: size, Matrix, show, display, *, +, /, isapprox, getindex

import Compat.LinearAlgebra: diag, eigvals, det, transpose
using Compat.LinearAlgebra: UpperTriangular, isdiag, Diagonal
import Compat: tr

if VERSION >= v"0.7"
    import LinearAlgebra: LinearAlgebra, adjoint, Adjoint, mul!

    A_mul_Bt!(Y, A, B) = mul!(Y, A, transpose(B))
else
    import Base: Ac_mul_B, A_mul_Bc
    import Base: ctranspose
    const adjoint = ctranspose
    const mul! = Base.A_mul_B!
end

"""
    @linalg_compat <function definition>

Given a method definition for `Ac_mul_B` or `A_mul_Bc`, generate a corresponding method for
`*` on Julia 0.7+ which is defined in terms of the original method definition.

For example, a method `Ac_mul_B(b::BlockDiagonal, m::BlockDiagonal)` will generate a
companion method
`(*)(b::Adjoint{BlockDiagonal}, m::BlockDiagonal) = Ac_mul_B(parent(b), m)`.

At some point these methods should be rewritten in terms of * on `Adjoint` types, so this
macro can be purged from existence.

!!! note

    This macro currently requires a long-form unparameterized method definition where the
    types of both arguments are specified.
"""
macro linalg_compat(expr)
    if VERSION < v"0.7"
        return esc(expr)
    end

    e = ArgumentError("Function definition did not match format expected by @linalg_compat")

    # aggressive input checking to replace the MacroTools matcher
    if expr.head !== :function || length(expr.args) != 2
        throw(e)
    end

    sig, body = expr.args

    if sig.head !== :call || length(sig.args) != 3
        throw(e)
    end

    f = sig.args[1]

    if sig.args[2].head !== :(::) || length(sig.args[2].args) != 2 ||
        sig.args[3].head !== :(::) || length(sig.args[3].args) != 2
        throw(e)
    end
    arg1, Type1 = sig.args[2].args
    arg2, Type2 = sig.args[3].args

    if f === :Ac_mul_B
        Type1 = :(Adjoint{$Type1})
        reassignment = :($arg1 = parent($arg1))
    elseif f === :A_mul_Bc
        Type2 = :(Adjoint{$Type2})
        reassignment = :($arg2 = parent($arg2))
    else
        error("Unrecognized method $f")
    end

    return quote
        function Base.:(*)(($(arg1))::($(Type1)), ($(arg2))::($(Type2)))
            $reassignment
            $(f)($(arg1), $(arg2))
        end
        $(esc(expr))
    end
end

struct BlockDiagonal{T} <: AbstractMatrix{T}
    blocks::Vector{<:AbstractMatrix{T}}
end

blocks(b::BlockDiagonal) = b.blocks
diag(b::BlockDiagonal) = vcat(diag.(blocks(b))...)

function size(b::BlockDiagonal)
    sizes = size.(blocks(b))
    return sum.(([s[1] for s in sizes], [s[2] for s in sizes]))
end

function isapprox(b1::BlockDiagonal, b2::BlockDiagonal; atol::Real=0)
    size(b1) != size(b2) && return false
    !all(size.(blocks(b1)) == size.(blocks(b2))) && return false
    return all(isapprox.(blocks(b1), blocks(b2), atol=atol))
end
function isapprox(b1::BlockDiagonal, b2::AbstractMatrix; atol::Real=0)
    return isapprox(Matrix(b1), b2, atol=atol)
end
function isapprox(b1::AbstractMatrix, b2::BlockDiagonal; atol::Real=0)
    return isapprox(b1, Matrix(b2), atol=atol)
end

Matrix(b::BlockDiagonal) = cat(blocks(b)...; dims=(1, 2))

chol(b::BlockDiagonal) = BlockDiagonal(chol.(blocks(b)))
det(b::BlockDiagonal) = prod(det.(blocks(b)))
function eigvals(b::BlockDiagonal)
    eigs = vcat(eigvals.(blocks(b))...)
    !isa(eigs, Vector{<:Complex}) && return sort(eigs)
    return eigs
end
tr(b::BlockDiagonal) = sum(tr.(blocks(b)))

transpose(b::BlockDiagonal) = BlockDiagonal(transpose.(blocks(b)))
adjoint(b::BlockDiagonal) = BlockDiagonal(adjoint.(blocks(b)))

function getindex(b::BlockDiagonal{T}, i::Int, j::Int) where T
    cols = [size(bb, 2) for bb in blocks(b)]
    rows = [size(bb, 1) for bb in blocks(b)]
    c = 0
    while j > 0
        c += 1
        j -= cols[c]
    end
    i = i - sum(rows[1:(c - 1)])
    (i <= 0 || i > rows[c]) && return zero(T)
    return blocks(b)[c][i, end + j]
end

function (*)(b::BlockDiagonal, m::AbstractMatrix)
    size(b, 2) != size(m, 1) && throw(
        DimensionMismatch("A has dimensions $(size(b)) but B has dimensions $(size(m))")
    )
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 2) - 1
        push!(d, block * m[st:ed, :])
        st = ed + 1
    end
    return vcat(d...)
end
function (*)(m::AbstractMatrix, b::BlockDiagonal)
    size(b, 1) != size(m, 2) && throw(
        DimensionMismatch("A has dimensions $(size(b)) but B has dimensions $(size(m))")
    )
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 1) - 1
        push!(d, m[:, st:ed] * block)
        st = ed + 1
    end
    return hcat(d...)
end
function (*)(b::BlockDiagonal, m::BlockDiagonal)
    if size(b) == size(m) && size.(blocks(b)) == size.(blocks(m))
        return BlockDiagonal(blocks(b) .* blocks(m))
    else
        Matrix(b) * Matrix(m)
    end
end
@linalg_compat function Ac_mul_B(b::BlockDiagonal, m::BlockDiagonal)
    if size(b) == size(m) && size.(blocks(b)) == size.(blocks(m))
        return BlockDiagonal(Ac_mul_B.(blocks(b), blocks(m)))
    else
        Ac_mul_B(Matrix(b), Matrix(m))
    end
end
@linalg_compat function Ac_mul_B(b::BlockDiagonal, m::AbstractMatrix)
    size(b, 1) != size(m, 1) && throw(
        DimensionMismatch("A has dimensions $(size(b)) but B has dimensions $(size(m'))")
    )
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 1) - 1
        push!(d, Ac_mul_B(block, m[st:ed, :]))
        st = ed + 1
    end
    return vcat(d...)
end
@linalg_compat function Ac_mul_B(m::AbstractMatrix, b::BlockDiagonal)
    size(b, 1) != size(m, 1) && throw(
        DimensionMismatch("A has dimensions $(size(b)) but B has dimensions $(size(m'))")
    )
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 1) - 1
        push!(d, Ac_mul_B(m[st:ed, :], block))
        st = ed + 1
    end
    return hcat(d...)
end
@linalg_compat function A_mul_Bc(b::BlockDiagonal, m::BlockDiagonal)
    if size(b) == size(m) && size.(blocks(b)) == size.(blocks(m))
        return BlockDiagonal(A_mul_Bc.(blocks(b), blocks(m)))
    else
        A_mul_Bc(Matrix(b), Matrix(m))
    end
end
@linalg_compat function A_mul_Bc(b::BlockDiagonal, m::AbstractMatrix)
    size(b, 2) != size(m, 2) && throw(
        DimensionMismatch("A has dimensions $(size(b)) but B has dimensions $(size(m'))")
    )
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 2) - 1
        push!(d, A_mul_Bc(block, m[:, st:ed]))
        st = ed + 1
    end
    return vcat(d...)
end
@linalg_compat function A_mul_Bc(m::AbstractMatrix, b::BlockDiagonal)
    size(b, 2) != size(m, 2) && throw(
        DimensionMismatch("A has dimensions $(size(b)) but B has dimensions $(size(m'))")
    )
    st = 1
    ed = 1
    d = []
    for block in blocks(b)
        ed = st + size(block, 2) - 1
        push!(d, A_mul_Bc(m[:, st:ed], block))
        st = ed + 1
    end
    return hcat(d...)
end
(*)(b::BlockDiagonal, n::Real) = BlockDiagonal(n .* blocks(b))
(*)(n::Real, b::BlockDiagonal) = b * n

(/)(b::BlockDiagonal, n::Real) = BlockDiagonal(blocks(b) ./ n)

function (+)(b::BlockDiagonal, m::AbstractMatrix)
    !isdiag(m) && return Matrix(b) + m
    size(b) != size(m) && throw(DimensionMismatch("Can't add matrices of different sizes."))
    d = diag(m)
    si = 1
    sj = 1
    nb = copy(blocks(b))
    for i in 1:length(nb)
        s = size(nb[i])
        nb[i] += @view m[si:s[1] + si - 1, sj:s[2] + sj - 1]
        si += s[1]
        sj += s[2]
    end
    return BlockDiagonal(nb)
end

(+)(m::AbstractMatrix, b::BlockDiagonal) = b + m
function (+)(b::BlockDiagonal, m::BlockDiagonal)
    if size(b) == size(m) && size.(blocks(b)) == size.(blocks(m))
        return BlockDiagonal(blocks(b) .+ blocks(m))
    else
        return Matrix(b) + Matrix(m)
    end
end

"""
    outer(x::Array, y::Array) -> Matrix
    outer(x::Array) -> Matrix

Compute outer product between `x` and `y` or between `x` and itself.
"""
outer(x::Array, y::Array) = x * y'
outer(x::Array) = outer(x, x)

function check_sizes(A::AbstractMatrix, B::AbstractMatrix)
    (k, l), (m, n) = size(A), size(B)
    eye_n, rem = divrem(m, l)
    rem > 0 && throw(DimensionMismatch("Sizes of the inputs are incompatible."))
    return (k, l), (m, n), eye_n
end

"""
    kron_lid_lmul(A::AbstractMatrix, B::AbstractMatrix) -> Matrix

Efficiently compute `kron(eye(n), A) * B`, assuming that such `n` exists.
"""
function kron_lid_lmul(A::AbstractMatrix, B::AbstractMatrix)
    (k, l), (m, n), eye_n = check_sizes(A, B)
    return reshape(A * reshape(B, l, div(m * n, l)), eye_n * k, n)
end

"""
    kron_lid_lmul_lt_s(A::AbstractMatrix, B::AbstractMatrix) -> Matrix

Efficiently compute `kron(eye(n), A) * B`, assuming that such `n` exists. For B lower
triangular. Optimised for execution speed.
"""
function kron_lid_lmul_lt_s(A::AbstractMatrix, B::AbstractMatrix)
    (k, l), (m, n), eye_n = check_sizes(A, B)
    blocks = Vector{Matrix{Float64}}(undef, eye_n)
    for i in 1:eye_n
        blocks[i] = create_block(A, B, k, l, m, n, i)
    end
    return vcat(blocks...)
end

function create_block(A, B, k, l, m, n, i)
    hcat(A * B[l*(i-1)+1:i*l, 1:(i*l)], zeros(Float64, (k, n-l*i)))
end

"""
    kron_lid_lmul_lt_m(A::AbstractMatrix, B::AbstractMatrix) -> Matrix

Efficiently compute `kron(eye(n), A) * B`, assuming that such `n` exists. For B lower
triangular. Optmised for memory.
"""
function kron_lid_lmul_lt_m(A::AbstractMatrix, B::AbstractMatrix)
    (k, l), (m, n), eye_n = check_sizes(A, B)
    res = Matrix{Float64}(undef, eye_n * k, n)
    for i = 1:eye_n;
        mul!(view(res, (i - 1) * k + 1:i * k, :), A, view(B, (i - 1) * l + 1: i * l, :))
    end
    return res
end

"""
    kron_rid_lmul_s(A::AbstractMatrix, B::AbstractMatrix) -> Matrix

Efficiently compute `kron(A, eye(n)) * B`, assuming that such `n` exists. Optimised for
execution time.
"""
function kron_rid_lmul_s(A::AbstractMatrix, B::AbstractMatrix)
    (k, l), (m, n), eye_n = check_sizes(A, B)
    return hcat(
        [reshape(reshape(B[:, i], div(m * 1, l), l) * A', eye_n * k, 1) for i = 1:n]...
    )
end

"""
    kron_rid_lmul_m(A::AbstractMatrix, B::AbstractMatrix) -> Matrix

Efficiently compute `kron(A, eye(n)) * B`, assuming that such `n` exists. Optimised for
memory.
"""
function kron_rid_lmul_m(A::AbstractMatrix, B::AbstractMatrix)
    (k, l), (m, n), eye_n = check_sizes(A, B)
    C = Matrix{Float64}(undef, k * eye_n, n)
    for i = 1:n
        @inbounds A_mul_Bt!(
            reshape(view(C, :, i), eye_n, k),
            reshape(view(B, :, i), eye_n, l),
            A
        )
    end
    return C
end

"""
    kron_lmul_rl(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix) -> Matrix

Efficiently compute `kron(A, B) * C`, writing it as `kron(A, eye(n)) * kron(eye(n), B) * C`.
Should be preferred when A is larger.
"""
function kron_lmul_rl(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    return kron_rid_lmul_m(A, kron_lid_lmul(B, C))
end

"""
    kron_lmul_lr(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix) -> Matrix

Efficiently compute `kron(A, B) * C`, writing it as `kron(eye(n), B) * kron(A, eye(n)) * C`.
Should be preferred when B is larger.
"""
function kron_lmul_lr(A::AbstractMatrix, B::AbstractMatrix, C::AbstractMatrix)
    return kron_lid_lmul(B, kron_rid_lmul_m(A, C))
end

"""
    diag_outer_kron(A::AbstractMatrix, B::AbstractMatrix) -> Matrix

Efficiently compute `diag(outer(kron(A, B)))`.
"""
function diag_outer_kron(A::AbstractMatrix, B::AbstractMatrix)
    kron(sum(A .^ 2, dims=2), sum(B .^ 2, dims=2))
end

"""
    diag_outer_kron(
        A::AbstractArray,
        B::AbstractArray,
        C::AbstractArray,
        D::AbstractArray
    ) -> Matrix

Efficiently compute `diag(outer(kron(A, B), kron(C, D)))`.
"""
function diag_outer_kron(
    A::AbstractArray,
    B::AbstractArray,
    C::AbstractArray,
    D::AbstractArray
)
    return kron(sum(A .* C, dims=2), sum(B .* D, dims=2))
end

"""
    diag_outer(A::AbstractArray, B::AbstractArray) -> Matrix

Efficiently compute `diag(outer(A, B))`.
"""
diag_outer(A::AbstractArray, B::AbstractArray) = sum(A .* B, dims=2)

"""
    diag_outer(A::AbstractMatrix)

Efficiently compute `diag(outer(A))`.
"""
diag_outer(A::AbstractArray) = sum(A .^ 2, dims=2)

"""
    Js(m::Int) -> Vector{Matrix}

Build an array of `mxm` matrices containing the `m` possible matrices in which only a single
diagonal element is non-zero and equal to one.
"""
function Js(m::Int)
    J = [zeros(m) for i = 1:m]
    for i = 1:m
        J[i][i] = 1
    end
    return Diagonal.(J)
end

"""
    Ms(m::Int) -> Vector{Vector{Matrix}}

Build an array of `m` arrays of `mxm` matrices containing the `m` possible matrices in which
only the ith entry of the jth line is non-zero and equal to one.
"""
function Ms(m::Int)
    M = [[zeros(m, m) for i = 1:m] for j = 1:m]
    for i = 1:m, j = 1:m
        M[i][j][i, j] = 1
    end
    return M
end

const NA = Nabla.∇Array
const Mat{T} = AbstractArray{T, 2} where N
"""
    sum_kron_J_ut(m::Integer, K::NA...)

Efficiently compute and differentiate `sum([kron(L[i], J[i]) for i = 1:m])` where `L` are
upper triangular.
"""
function sum_kron_J_ut(m::Integer, K::NA...)
    n = size(K[1], 1)
    res = zeros(m * n, m * n)
    @inbounds for i = 1:m
        for k = 1:n
            for l = k:n
                res[i + (k - 1) * m, i + (l - 1) * m] = K[i][k, l]
            end
        end
    end
    return res
end
@union_intercepts sum_kron_J_ut Tuple{Integer, Vararg{NA}} Tuple{Integer, Vararg{NA}}
function Nabla.∇(::typeof(sum_kron_J_ut), ::Type{Arg{i}}, _, y, ȳ, m, K...) where i
    # TODO: Check this is okay!
    n = size(ȳ, 1)
    return view(ȳ, i - 1:m:n, i - 1:m:n)
end

#TODO: provide a similar implementation of sum_kron_M that computes
# sum([kron(K[i, j], M[i][j]) for i = 1:m1 for j = 1:m2]) for the multikernels.
"""
    eye_sum_kron_M_ut(B::Mat{T}, L::UpperTriangular{T}...) where T

Efficiently compute and differentiate the upper triangle of
`eye(n * m) + sum([kron(L[i] * L[j]', B[i, j] * M[i][j]) for i = 1:m for j = 1:m])`.
"""
function eye_sum_kron_M_ut(B::Mat{T}, L::UpperTriangular{T}...) where T
    # Assumes that `ȳ` is upper triangular.
    m, n = size(B, 1), size(L[1], 1)
    res = Matrix{T}(undef, m * n, m * n)
    L_prod = Matrix{T}(undef, n, n)
    for i = 1:m, j = i:m
        A_mul_Bt!(L_prod, L[i], L[j])
        _eskmu_fill_triu!(res, n, m, i, j, B, L_prod)
        i == j ? _eskmu_fill_diag!(res, n, m, i) :
                 _eskmu_fill_triu_Lt!(res, n, m, j, i, B, L_prod)
    end
    return res
end
function _eskmu_fill_diag!(res, n, m, i)
    @simd for k = 1:n
        @inbounds res[i + (k - 1) * m, i + (k - 1) * m] += 1
    end
end
function _eskmu_fill_triu!(res, n, m, i, j, B, L_prod)
    for k = 1:n
        @simd for l = k:n
            @inbounds res[i + (k - 1) * m, j + (l - 1) * m] = B[i, j] * L_prod[k, l]
        end
    end
end
function _eskmu_fill_triu_Lt!(res, n, m, i, j, B, L_prod)
    for k = 1:n
        @simd for l = k:n
            @inbounds res[i + (k - 1) * m, j + (l - 1) * m] = B[i, j] * L_prod[l, k]
        end
    end
end

@union_intercepts eye_sum_kron_M_ut Tuple{NA, Vararg{NA}} Tuple{NA, Vararg{NA}}
function Nabla.∇(
    ::typeof(eye_sum_kron_M_ut),
    ::Type{Arg{1}},
    _,
    y::Mat{T},
    ȳ::Mat{T},
    B::Mat{T},
    L::UpperTriangular{T}...
) where T
    m, n = size(B, 1), size(L[1], 1)
    B̄ = zeros(T, size(B))
    L_prod = Matrix{T}(n, n)
    @inbounds for i = 1:m, j = 1:m
        A_mul_Bt!(L_prod, L[i], L[j])
        _esku_sum!(B̄, m, n, i, j, ȳ, L_prod)
    end
    return B̄
end
function _esku_sum!(B̄, m, n, i, j, ȳ, L_prod)
    b̄ij = 0.0
    for k = 1:n
        @simd for l = k:n
            @inbounds b̄ij += ȳ[(k - 1) * m + i, (l - 1) * m + j] * L_prod[k, l]
        end
    end
    B̄[i, j] = b̄ij
end

function Nabla.∇(
    ::typeof(eye_sum_kron_M_ut),
    ::Type{Arg{i}},
    _,
    y::Mat{T},
    ȳ::Mat{T},
    B::Mat{T},
    L::UpperTriangular{T}...
) where i where T
    # Assumes that `ȳ` is upper triangular and that `B` is symmetric.
    s, m, n = i - 1, size(B, 1), size(L[1], 1)
    L̄s = zeros(T, size(L[s]))
    ȳ = UpperTriangular(ȳ)
    # This is what is implemented:
    # L̄s[i, j] += ȳ[(i - 1) * m + s, (k - 1) * m + l] * B[s, l] * L[l][k, j] +
    #             ȳ[(k - 1) * m + l, (i - 1) * m + s] * B[l, s] * L[l][k, j]
    for l = 1:m
        @inbounds bsl = B[s, l]
        @inbounds L̄s .+= bsl .* (view(ȳ, s:m:n * m, l:m:n * m) * L[l])
        @inbounds L̄s .+= bsl .* At_mul_B(view(ȳ, l:m:n * m, s:m:n * m), L[l])
    end
    return L̄s
end

end # module end
