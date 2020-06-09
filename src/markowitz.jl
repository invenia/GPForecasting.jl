@unionise function PO_analytical(μ, Σ, A, b)
    L = cholesky(Σ).L
    Z = L \ A'
    z = L \ μ
    w = 0.5 * (L' \ (L \ (μ + A'*((Z'*Z) \ (b - Z'*z)))))
    return w
end

# 8.2 Total negative log likelihood of validation set
function lldeltas(
        gp::GP{K, M},
        x,
        y,
) where {K <: OLMMKernel, M <: Mean}

    s = 0

    for i=1:size(x, 1)
        μ = gp.m(x[i:i, :])[:]
        Σ = Symmetric(gp.k(x[i:i, :]))

        L = cholesky(Σ).L
        log_det = 2sum(log, diag(L))
        vy = y[i, :] - μ
        z = L \ vy
        s -= 0.5 * (log_det + length(vy) * log(2π) + sum(abs2, z))
    end

    return s
end

function lldeltas_obj(gp::GP, x_train, x_val, y_train, y_val, w_train, w_val)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, x_train, y_train)
        obj = -lldeltas(pos::GP, x_val, y_val)
        return obj
    end
end

# 9.1 Total return of validation set
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
        w /= norm(w)  # normalization is required to have a bounded objective
        s += dot(w, y[i, :])
    end
    return s
end

function totalreturn_obj(gp::GP, x_train, x_val, y_train, y_val, w_train, w_val)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, x_train, y_train)
        obj = -totalreturn(pos, x_val, y_val)
        return obj
    end
end

# 9.3 MSE of predicted returns of validation set
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
        r_pred = dot(w_pred, μ)
        r_true = dot(w[i, :], y[i, :]) # TODO: Calculate prior to this
        s += (r_pred - r_true)^2
    end
    return s / size(x, 1)
end

function msereturns_obj(gp::GP, x_train, x_val, y_train, y_val, w_train, w_val)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, x_train, y_train)
        obj = msereturns(pos, x_val, y_val, w_val)
        return obj
    end
end

# 9.4 Total negative log likelihood of predicted returns of validation set
function llreturns(
        gp::GP{K, M},
        x,
        y,
        w,
) where {K <: OLMMKernel, M <: Mean}

    A = ones(size(y)[2])'
    b = 0

    s = 0

    for i=1:size(x, 1)
        μ = gp.m(x[i:i, :])[:]
        Σ = Symmetric(gp.k(x[i:i, :]))
        w_pred = GPForecasting.PO_analytical(μ, Σ, A, b)
        # Translate the price distribution into the returns distribution through affine transformation
        \mu_r_pred = dot(w_pred, μ)
        σ² = dot(w_pred, Σ*w_pred)
        r_true = dot(w[i, :], y[i, :])
        σ² = dot(w_pred, Σ*w_pred)
        s -= 0.5 * (log(σ²) + (r_pred - r_true)^2 / σ² + log(2π))
    end

    return s
end

function llreturns_obj(gp::GP, x_train, x_val, y_train, y_val, w_train, w_val)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, x_train, y_train)
        obj = -llreturns(pos, x_val, y_val, w_val)
        return obj
    end
end

# 9.7 MSE of the weights of validation set
function msevolumes(
        gp::GP{K, M},
        x,
        y,
        w,
) where {K <: OLMMKernel, M <: Mean}

    A = ones(size(y)[2])'
    b = 0

    s = 0

    for i=1:size(x, 1)
        μ = gp.m(x[i:i, :])[:]
        Σ = Symmetric(gp.k(x[i:i, :]))
        w_pred = GPForecasting.PO_analytical(μ, Σ, A, b)
        s += mean((w_pred - w[i, :]).^2)
    end

    return s / size(x, 1)
end

function msevolumes_obj(gp, x_train, x_val, y_train, y_val, w_train, w_val)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, x_train, y_train)
        obj = msevolumes(pos, x_val, y_val, w_val)
        return obj
    end
end
