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
    b = 0

    # TODO: Explore whether defining these in one call or individually is more performant
    # COMMENT: Looks like many changes would be needed for this, mostly around
    # supporting Nabla
    s = 0

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
    b = 0

    # TODO: Explore whether defining these in one call or individually is more performant
    # COMMENT: Looks like many changes would be needed for this, mostly around
    # supporting Nabla
    s = 0

    for i=1:size(x, 1)
        μ = gp.m(x[i:i, :])[:]
        Σ = Symmetric(gp.k(x[i:i, :]))
        w_pred = PO_analytical(μ, Σ, A, b)
        r_true = dot(w[i,:], y[i,:]) # TODO: Calculate prior to this
        s += (dot(w_pred, y[i, :]) - r_true)^2
    end
    return s / size(x, 1)
end

# 9.1
function totalreturn_obj(gp::GP, x_train, x_val, y_train, y_val, w_train, w_val)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, x_train, y_train)
        obj = -totalreturn(pos, x_val, y_val)
        return obj
    end
end

# 9.3
function msereturns_obj(gp::GP, x_train, x_val, y_train, y_val, w_train, w_val)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, x_train, y_train)
        obj = msereturns(pos::GP, x_val, y_val, w_val)
        return obj
    end
end

