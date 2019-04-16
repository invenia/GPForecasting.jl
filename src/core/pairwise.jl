@unionise function pairwise_dist(x::AbstractArray, y::AbstractArray)
    # based on GPflow implementation
    return sqrt.(max.(sq_pairwise_dist(x, y), 1.0e-40))
end

@unionise pairwise_dist(x::Number, y::AbstractArray) = pairwise_dist([x], y)
@unionise pairwise_dist(x::AbstractArray, y::Number) = pairwise_dist(x, [y])
@unionise pairwise_dist(x::Number, y::Number) = pairwise_dist([x], [y])

@unionise function sq_pairwise_dist(x::AbstractArray, y::AbstractArray)
    return sum(abs2, x, dims=2) .+ sum(abs2, y, dims=2)' .- 2x * y'
end

@unionise sq_pairwise_dist(x::Number, y::AbstractArray) = sq_pairwise_dist([x], y)
@unionise sq_pairwise_dist(x::AbstractArray, y::Number) = sq_pairwise_dist(x, [y])
@unionise sq_pairwise_dist(x::Number, y::Number) = sq_pairwise_dist([x], [y])
