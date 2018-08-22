export cov_EB, cov_LW 


"""
    cov_EB(X::AbstractArray)

Return regularised covariance matrix of matrix `X` using simple Empirical Bayesian approach
"""
function cov_EB(X)
    n = size(X)[1]
    p = size(X)[2]

    S = cov(X, 1, false)
    m = trace(S) / p
    return ((p * n - 2 * n - 2) / (p * n^2) * m) .* eye(S) .+ (n / (n+1)) .* S
end

"""
    cov_LW(X::AbstractArray)

Return regularised covariance matrix of matrix `X` using the Ledoit-Wolf approach 
"""
function cov_LW(X)
    n = size(X)[1]
    p = size(X)[2]

    S = cov(X, 1, false)
    Y = X .- mean(X, 1)
    m = trace(S) / p
    d2 = vecnorm(S .- m .* eye(S))^2 / p
    b2 = 0.0
    for k=1:n
        y = Y[k,:]
        b2 += vecnorm(y * y' .- S)^2
    end
    b2 = b2 / n^2 / p
    b2 = min(b2, d2)
    if b2 == d2
        warn("a2 is zero => covariance matrix is scaled identity matrix")
    end
    a2 = d2 - b2
    return (b2 / d2 * m) .* eye(S) .+ (a2 / d2) .* S
end
