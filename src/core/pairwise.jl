export pairwise_dist, sq_pairwise_dist

@unionise function pairwise_dist(x::AbstractArray, y::AbstractArray)
    # based on GPflow implementation
    return sqrt.(max.(sq_pairwise_dist(x, y), 1.0e-40))
end

@unionise pairwise_dist(x::Number, y::AbstractArray) = pairwise_dist([x], y)
@unionise pairwise_dist(x::AbstractArray, y::Number) = pairwise_dist(x, [y])
@unionise pairwise_dist(x::Number, y::Number) = pairwise_dist([x], [y])

# This function is very inneficient and is only used as a temporary work-around a Nabla
# issue. Once that is solved, we should move back to using sumdims(x.^2, 2)
@unionise function mydumbsum(x::AbstractArray)
    xs = [x[:, i] for i in 1:size(x, 2)]
    out = zeros(size(x, 1))
    for i in 1:size(x, 2)
       out += xs[i]
    end
    return out
end

@unionise function sq_pairwise_dist(x::AbstractArray, y::AbstractArray)
    return sumdims(x.^2, 2) .+ sumdims(y.^2, 2)' .- 2x * y'
end

@unionise sq_pairwise_dist(x::Number, y::AbstractArray) = sq_pairwise_dist([x], y)
@unionise sq_pairwise_dist(x::AbstractArray, y::Number) = sq_pairwise_dist(x, [y])
@unionise sq_pairwise_dist(x::Number, y::Number) = sq_pairwise_dist([x], [y])
