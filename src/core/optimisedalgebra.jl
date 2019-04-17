module OptimisedAlgebra

using LinearAlgebra
using Nabla

export outer, kron_lid_lmul, kron_lid_lmul_lt_s, kron_lid_lmul_lt_m,
kron_rid_lmul_s, kron_rid_lmul_m, kron_lmul_lr, kron_lmul_rl, diag_outer_kron,
diag_outer, Js, Ms, sum_kron_J_ut, sum_kron_M, eye_sum_kron_M_ut, BlockDiagonal, blocks


struct BlockDiagonal{T} <: AbstractMatrix{T}
    blocks::Vector{<:AbstractMatrix{T}}
end

blocks(b::BlockDiagonal) = b.blocks
LinearAlgebra.diag(b::BlockDiagonal) = mapreduce(diag, vcat, blocks(b))

function Base.size(b::BlockDiagonal)
    sizes = size.(blocks(b))
    return sum.(([s[1] for s in sizes], [s[2] for s in sizes]))
end

function Base.isapprox(b1::BlockDiagonal, b2::BlockDiagonal; atol::Real=0)
    size(b1) != size(b2) && return false
    !all(size.(blocks(b1)) == size.(blocks(b2))) && return false
    return all(isapprox.(blocks(b1), blocks(b2), atol=atol))
end
function Base.isapprox(b1::BlockDiagonal, b2::AbstractMatrix; atol::Real=0)
    return isapprox(Matrix(b1), b2, atol=atol)
end
function Base.isapprox(b1::AbstractMatrix, b2::BlockDiagonal; atol::Real=0)
    return isapprox(b1, Matrix(b2), atol=atol)
end

Base.Matrix(b::BlockDiagonal) = cat(blocks(b)...; dims=(1, 2))

function LinearAlgebra.cholesky(B::BlockDiagonal)
    return Cholesky(BlockDiagonal(map(b->cholesky(b).U, blocks(B))), 'U', 0)
end

# Make getproperty on a Cholesky factorized BlockDiagonal return another BlockDiagonal
# where each block is an upper or lower triangular matrix. This ensures that optimizations
# for BlockDiagonal matrices are preserved, though it comes at the cost of reallocating
# a vector of triangular wrappers on each call.
function Base.getproperty(C::Cholesky{T,BlockDiagonal{T}}, x::Symbol) where T
    B = getfield(C, :factors)
    uplo = getfield(C, :uplo)
    if x === :U
        f = uplo === 'U' ? UpperTriangular : (X->UpperTriangular(X'))
    elseif x === :L
        f = uplo === 'L' ? LowerTriangular : (X->LowerTriangular(X'))
    elseif x === :UL
        f = uplo === 'U' ? UpperTriangular : LowerTriangular
    else
        return getfield(C, x)
    end
    return BlockDiagonal{T}(map(f, blocks(B)))
end

LinearAlgebra.det(b::BlockDiagonal) = prod(det.(blocks(b)))
function LinearAlgebra.eigvals(b::BlockDiagonal)
    eigs = mapreduce(eigvals, vcat, blocks(b))
    !isa(eigs, Vector{<:Complex}) && return sort(eigs)
    return eigs
end
LinearAlgebra.tr(b::BlockDiagonal) = sum(tr.(blocks(b)))

LinearAlgebra.transpose(b::BlockDiagonal) = BlockDiagonal(transpose.(blocks(b)))
LinearAlgebra.adjoint(b::BlockDiagonal) = BlockDiagonal(adjoint.(blocks(b)))

function Base.getindex(b::BlockDiagonal{T}, i::Int, j::Int) where T
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

function Base.:*(b::BlockDiagonal, m::AbstractMatrix)
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
    return reduce(vcat, d)
end
function Base.:*(m::AbstractMatrix, b::BlockDiagonal)
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
    return reduce(hcat, d)
end
function Base.:*(b::BlockDiagonal, m::BlockDiagonal)
    if size(b) == size(m) && size.(blocks(b)) == size.(blocks(m))
        return BlockDiagonal(blocks(b) .* blocks(m))
    else
        return Matrix(b) * Matrix(m)
    end
end
Base.:*(b::BlockDiagonal, n::Real) = BlockDiagonal(n .* blocks(b))
Base.:*(n::Real, b::BlockDiagonal) = b * n

Base.:/(b::BlockDiagonal, n::Real) = BlockDiagonal(blocks(b) ./ n)

function Base.:+(b::BlockDiagonal, m::AbstractMatrix)
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

Base.:+(m::AbstractMatrix, b::BlockDiagonal) = b + m
function Base.:+(b::BlockDiagonal, m::BlockDiagonal)
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
    return reduce(vcat, blocks)
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
    cols = map(1:n) do i
        C = reshape(B[:, i], (:, l)) * A'
        reshape(C, (:, 1))
    end
    return reduce(hcat, cols)
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
        @inbounds mul!(
            reshape(view(C, :, i), eye_n, k),
            reshape(view(B, :, i), eye_n, l),
            transpose(A)
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
    kron(sum(abs2, A, dims=2), sum(abs2, B, dims=2))
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
diag_outer(A::AbstractArray) = sum(abs2, A, dims=2)

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
function sum_kron_J_ut end

function _sum_kron_J_ut(m::Integer, K::NA...)
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
# @union_intercepts sum_kron_J_ut Tuple{Integer, Vararg{NA}} Tuple{Integer, Vararg{NA}}
# manually expanded to avoid stack overflow:
@generated function sum_kron_J_ut(x1::Union{Integer, Node{<:Integer}}, x2::Vararg{Union{NA, Node{<:NA}}})
    x = ([x1, x2]...,)
    x_syms = ([:x1, :x2]...,)
    x_dots = (:x1, Expr(:(...), :x2))
    is_node = [any((<:).(xj, Node)) for xj = x]
    if any(is_node)
        Nabla.branch_expr(:_sum_kron_J_ut, is_node, x, x_syms, :((x1, x2...)))
    else
        Expr(:call, :_sum_kron_J_ut, x_dots...)
    end
end
function Nabla.∇(::typeof(_sum_kron_J_ut), ::Type{Arg{i}}, _, y, ȳ, m, K...) where i
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
function eye_sum_kron_M_ut end

function _eye_sum_kron_M_ut(B::Mat{T}, L::UpperTriangular{T}...) where T
    # Assumes that `ȳ` is upper triangular.
    m, n = size(B, 1), size(L[1], 1)
    res = Matrix{T}(undef, m * n, m * n)
    L_prod = Matrix{T}(undef, n, n)
    for i = 1:m, j = i:m
        mul!(L_prod, L[i], transpose(L[j]))
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

# @union_intercepts eye_sum_kron_M_ut Tuple{NA, Vararg{NA}} Tuple{NA, Vararg{NA}}
# manually expanded to avoid stack overflow:
@generated function eye_sum_kron_M_ut(x1::Union{NA, Node{<:NA}}, x2::Vararg{Union{NA, Node{<:NA}}})
    x = ([x1, x2]...,)
    x_syms = ([:x1, :x2]...,)
    x_dots = (:x1, Expr(:(...), :x2))
    is_node = [any((<:).(xj, Node)) for xj = x]
    if any(is_node)
        Nabla.branch_expr(:_eye_sum_kron_M_ut, is_node, x, x_syms, :((x1, x2...)))
    else
        Expr(:call, :_eye_sum_kron_M_ut, x_dots...)
    end
end
function Nabla.∇(
    ::typeof(_eye_sum_kron_M_ut),
    ::Type{Arg{1}},
    _,
    y::Mat{T},
    ȳ::Mat{T},
    B::Mat{T},
    L::UpperTriangular{T}...
) where T
    m, n = size(B, 1), size(L[1], 1)
    B̄ = zeros(T, size(B))
    L_prod = Matrix{T}(undef, n, n)
    @inbounds for i = 1:m, j = 1:m
        mul!(L_prod, L[i], transpose(L[j]))
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
    ::typeof(_eye_sum_kron_M_ut),
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
        @inbounds L̄s .+= bsl .* (transpose(view(ȳ, l:m:n * m, s:m:n * m)) * L[l])
    end
    return L̄s
end

end # module end
