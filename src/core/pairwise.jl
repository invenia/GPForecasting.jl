@unionise function pairwise_dist(x::AbstractArray, y::AbstractArray)
    # based on GPflow implementation
    return sqrt.(max.(sq_pairwise_dist(x, y), 1.0e-40))
end

@unionise pairwise_dist(x::Number, y::AbstractArray) = pairwise_dist([x], y)
@unionise pairwise_dist(x::AbstractArray, y::Number) = pairwise_dist(x, [y])
@unionise pairwise_dist(x::Number, y::Number) = pairwise_dist([x], [y])

function sq_pairwise_dist(x::AbstractMatrix, y::AbstractMatrix)
    return pairwise(SqEuclidean(), x, y, dims=1)
end

function sq_pairwise_dist(x::AbstractVector, y::AbstractVector)
    yᵀ = y'
    xyᵀ = x*yᵀ
    return @. abs2(x) + abs2(yᵀ) - 2*xyᵀ
end

sq_pairwise_dist(x::Number, y::AbstractArray) = sq_pairwise_dist([x], y)
sq_pairwise_dist(x::AbstractArray, y::Number) = sq_pairwise_dist(x, [y])
sq_pairwise_dist(x::Number, y::Number) = sq_pairwise_dist([x], [y])

@explicit_intercepts sq_pairwise_dist Tuple{Nabla.∇ArrayOrScalar, Nabla.∇ArrayOrScalar}

function Nabla.∇(
    ::typeof(sq_pairwise_dist),
    ::Type{Arg{1}},
    p, z, z̄,
    x::AbstractMatrix,
    y::AbstractMatrix,
)
    D = Diagonal(vec(sum(z̄, dims=2)))
    return 2*(D*x - z̄*y)
end

function Nabla.∇(
    ::typeof(sq_pairwise_dist),
    ::Type{Arg{2}},
    p, z, z̄,
    x::AbstractMatrix,
    y::AbstractMatrix,
)
    D = Diagonal(vec(sum(z̄, dims=1)))
    return 2*(D*y - z̄'x)
end

function Nabla.∇(
    ::typeof(sq_pairwise_dist),
    ::Type{Arg{i}},
    p, z, z̄,
    x::AbstractVector,
    y::AbstractVector,
) where i
    return ∇(sq_pairwise_dist, Arg{i}, p, z, z̄, reshape(x, :, 1), reshape(y, :, 1))
end

@unionise function elwise_dist(x::AbstractArray, y::AbstractArray)
    return colwise(Euclidean(), x', y')
end

@unionise function sq_elwise_dist(x::AbstractArray, y::AbstractArray)
    return colwise(SqEuclidean(), x', y')
end

@unionise elwise_dist(x::Number, y::AbstractArray) = euclidean(x, y...)
@unionise elwise_dist(x::AbstractArray, y::Number) = euclidean(x..., y)
@unionise elwise_dist(x::Number, y::Number) = euclidean(x, y)

@unionise sq_elwise_dist(x::Number, y::AbstractArray) = sqeuclidean(x, y...)
@unionise sq_elwise_dist(x::AbstractArray, y::Number) = sqeuclidean(x..., y)
@unionise sq_elwise_dist(x::Number, y::Number) = sqeuclidean(x, y)
