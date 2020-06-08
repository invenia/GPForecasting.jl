@unionise function PO_analytical(μ, Σ, A, b)
    L = cholesky(Σ).L
    Z = L \ A'
    z = L \ μ
    w = 0.5 * (L' \ (L \ (μ + A'*((Z'*Z) \ (b - Z'*z)))))
    return w
end

# TODO: REFACTOR TO EXPRESS IN TERMS OF RETURN

function totalreturn(
    gp::GP{K, M},
    x,
    y,
) where {K <: OLMMKernel, M <: Mean}

    A = ones(size(y)[2])'

    # TODO: Explore whether defining these in one call or individually is more performant
    # COMMENT: Looks like many changes would be needed for this, mostly around
    # supporting Nabla
    s = 0
    b = 0

    for i=1:size(x, 1)
        μ = gp.m(x[i:i, :])[:]
        Σ = Symmetric(gp.k(x[i:i, :]))
        w = PO_analytical(μ, Σ, A, b)
        w /= norm(w)
        s += dot(w, y[i, :])
    end
    return s
end

function msereturns(
    gp::GP{K, M},
    x,
    y,
    w,
) where {K <: OLMMKernel, M <: Mean}

    A = ones(size(y)[2])'

    # TODO: Explore whether defining these in one call or individually is more performant
    # COMMENT: Looks like many changes would be needed for this, mostly around
    # supporting Nabla
    se = 0
    b = 0

    N = size(x, 1)

    for i=1:N
        μ = gp.m(x[i:i, :])[:]
        Σ = Symmetric(gp.k(x[i:i, :]))
        w_pred = PO_analytical(μ, Σ, A, b)
        r_true = dot(w[i,:], y[i,:]) # TODO: Calculate prior to this
        se += (dot(w_pred, y[i, :]) - r_true)^2
    end
    return se / N
end

# UNTESTED
# function llreturns(
#     gp::GP{K, M},
#     x,
#     y,
#     w,
# ) where {K <: OLMMKernel, M <: Mean}
#
#     α = 1.0
#     A = ones(size(y)[2])'
#
#     # TODO: Explore whether defining these in one call or individually is more performant
#     # COMMENT: Looks like many changes would be needed for this, mostly around
#     # supporting Nabla
#     ll = 0
#     α = 1.0
#     b = 0
#
#     N = size(x, 1)
#
#     for i=1:N
#         μ = gp.m(x[i:i, :])[:]
#         Σ = Symmetric(gp.k(x[i:i, :]))
#         d = MvNormal(μ, Σ)
#         ll += d(r_true)
#     end
#     return ll
# end

# For these Markowitz objectives, in each case, we pass in the training
# and validation components of the features, prices and volumes respectively.
# Although not all objectives use the volumes, this makes the experimentation
# script simpler.

# 9.1
function totalreturn_obj(gp::GP, x_train, x_val, y_train, y_val, w_train, w_val)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, x_train, y_train)
        obj = -totalreturn(pos, x_val, y_val)
        return obj
    end
end

# # Presently breaking
# function mle_obj(gp::GP, x_train, x_val, y_train, y_val, w_train, w_val)
#     return function f(params)
#         ngp = GP(gp.m, set(gp.k, params))
#         pos = condition(ngp, x_train, y_train)
#         obj = -logpdf(pos::GP, x_val, y_val, params)
#         return obj
#     end
# end

# 9.3
function msereturns_obj(gp::GP, x_train, x_val, y_train, y_val, w_train, w_val)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, x_train, y_train)
        obj = msereturns(pos::GP, x_val, y_val, w_val)
        return obj
    end
end

# 9.4
# function llreturns_obj(gp::GP, x_train, x_val, y_train, y_val, w_train, w_val)
#     return function f(params)
#         ngp = GP(gp.m, set(gp.k, params))
#         pos = condition(ngp, x_train, y_train)
#         obj = -llreturns(pos::GP, x_val, y_val, w_val)
#         return obj
#     end
# end
