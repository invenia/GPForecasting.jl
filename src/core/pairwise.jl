export pairwise_dist, sq_pairwise_dist

@unionise function pairwise_dist(x::AbstractArray, y::AbstractArray)
    # based on GPflow implementation
    return sqrt.(max.(sq_pairwise_dist(x, y), 1.0e-40))
end

@unionise pairwise_dist(x::Number, y::AbstractArray) = pairwise_dist([x], y)
@unionise pairwise_dist(x::AbstractArray, y::Number) = pairwise_dist(x, [y])
@unionise pairwise_dist(x::Number, y::Number) = pairwise_dist([x], [y])[1, 1]

@unionise function sq_pairwise_dist(x::AbstractArray, y::AbstractArray)
    # return sum(x .^ 2, dims=2) .+ sum(y .^ 2, dims=2)' .- 2x * y'
    sum1 = sumdims(x .^ 2, 2)
    sum2 = adjoint(sumdims(y .^ 2, 2))
    term = 2x * y'
    return sum1 .+ sum2 .- term
end

@unionise sq_pairwise_dist(x::Number, y::AbstractArray) = sq_pairwise_dist([x], y)
@unionise sq_pairwise_dist(x::AbstractArray, y::Number) = sq_pairwise_dist(x, [y])
@unionise sq_pairwise_dist(x::Number, y::Number) = sq_pairwise_dist([x], [y])[1, 1]
