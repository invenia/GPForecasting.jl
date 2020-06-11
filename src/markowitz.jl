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

function lldeltas_obj(gp::GP, data_train::NamedTuple, data_val::NamedTuple)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, data_train.x, data_train.y)
        obj = -lldeltas(pos::GP, data_val.x, data_val.y)
        return obj
    end
end

function lldeltas(
        ef::EF,
        x,
        y,
)

    s = 0

    for i=1:size(x, 1)
        μ, Σ = ef(x[i:i, :])

        L = cholesky(Σ).L
        log_det = 2sum(log, diag(L))
        vy = y[i, :] - μ
        z = L \ vy
        s -= 0.5 * (log_det + length(vy) * log(2π) + sum(abs2, z))
    end

    return s
end

function lldeltas_obj(ef::EF, data_train::NamedTuple, data_val::NamedTuple)
    return function f(params)
        nef = EF(set(ef.k, params), ef.estimator, data_train.x, data_train.y)
        obj = -lldeltas(nef::EF, data_val.x, data_val.y)
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

function totalreturn_obj(gp::GP, data_train::NamedTuple, data_val::NamedTuple)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, data_train.x, data_train.y)
        obj = -totalreturn(pos, data_val.x, data_val.y)
        return obj
    end
end

function totalreturn(
    ef::EF,
    x,
    y,
)

    A = ones(size(y)[2])'
    b = 0

    # TODO: Explore whether defining these in one call or individually is more performant
    # COMMENT: Looks like many changes would be needed for this, mostly around
    # supporting Nabla
    s = 0

    for i=1:size(x, 1)
        μ, Σ = ef(x[i:i, :])
        w = PO_analytical(μ, Σ, A, b)
        w /= norm(w)  # normalization is required to have a bounded objective
        s += dot(w, y[i, :])
    end
    return s
end

function totalreturn_obj(ef::EF, data_train::NamedTuple, data_val::NamedTuple)
    return function f(params)
        nef = EF(set(ef.k, params), ef.estimator, data_train.x, data_train.y)
        obj = -totalreturn(nef, data_val.x, data_val.y)
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
        μ_r_pred = dot(w_pred, μ)
        r_true = dot(w[i, :], y[i, :]) # TODO: Calculate prior to this
        s += (μ_r_pred - r_true)^2
    end
    return s / size(x, 1)
end

function msereturns_obj(gp::GP, data_train::NamedTuple, data_val::NamedTuple)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, data_train.x, data_train.y)
        obj = msereturns(pos, data_val.x, data_val.y, data_val.w)
        return obj
    end
end

function msereturns(
    ef::EF,
    x,
    y,
    w,
)

    A = ones(size(y)[2])'
    b = 0

    # TODO: Explore whether defining these in one call or individually is more performant
    # COMMENT: Looks like many changes would be needed for this, mostly around
    # supporting Nabla
    s = 0

    for i=1:size(x, 1)
        μ, Σ = ef(x[i:i, :])
        w_pred = PO_analytical(μ, Σ, A, b)
        μ_r_pred = dot(w_pred, μ)
        r_true = dot(w[i, :], y[i, :]) # TODO: Calculate prior to this
        s += (μ_r_pred - r_true)^2
    end
    return s / size(x, 1)
end

function msereturns_obj(ef::EF, data_train::NamedTuple, data_val::NamedTuple)
    return function f(params)
        nef = EF(set(ef.k, params), ef.estimator, data_train.x, data_train.y)
        obj = msereturns(nef, data_val.x, data_val.y, data_val.w)
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
        μ_r_pred = dot(w_pred, μ)
        σ²_r_pred = dot(w_pred, Σ*w_pred)
        r_true = dot(w[i, :], y[i, :])
        s -= 0.5 * (log(σ²_r_pred) + (μ_r_pred - r_true)^2 / σ²_r_pred + log(2π))
    end

    return s
end

function llreturns_obj(gp::GP, data_train::NamedTuple, data_val::NamedTuple)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, data_train.x, data_train.y)
        obj = -llreturns(pos, data_val.x, data_val.y, data_val.w)
        return obj
    end
end

function llreturns(
        ef::EF,
        x,
        y,
        w,
)

    A = ones(size(y)[2])'
    b = 0

    s = 0

    for i=1:size(x, 1)
        μ, Σ = ef(x[i:i, :])
        w_pred = GPForecasting.PO_analytical(μ, Σ, A, b)
        # Translate the price distribution into the returns distribution through affine transformation
        μ_r_pred = dot(w_pred, μ)
        σ²_r_pred = dot(w_pred, Σ*w_pred)
        r_true = dot(w[i, :], y[i, :])
        s -= 0.5 * (log(σ²_r_pred) + (μ_r_pred - r_true)^2 / σ²_r_pred + log(2π))
    end

    return s
end

function llreturns_obj(ef::EF, data_train::NamedTuple, data_val::NamedTuple)
    return function f(params)
        nef = EF(set(ef.k, params), ef.estimator, data_train.x, data_train.y)
        obj = -llreturns(nef, data_val.x, data_val.y, data_val.w)
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

function msevolumes_obj(gp::GP, data_train::NamedTuple, data_val::NamedTuple)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, data_train.x, data_train.y)
        obj = msevolumes(pos, data_val.x, data_val.y, data_val.w)
        return obj
    end
end

function msevolumes(
        ef::EF,
        x,
        y,
        w,
)

    A = ones(size(y)[2])'
    b = 0

    s = 0

    for i=1:size(x, 1)
        μ, Σ = ef(x[i:i, :])
        w_pred = GPForecasting.PO_analytical(μ, Σ, A, b)
        s += mean((w_pred - w[i, :]).^2)
    end

    return s / size(x, 1)
end

function msevolumes_obj(ef::EF, data_train::NamedTuple, data_val::NamedTuple)
    return function f(params)
        nef = EF(set(ef.k, params), ef.estimator, data_train.x, data_train.y)
        obj = msevolumes(nef, data_val.x, data_val.y, data_val.w)
        return obj
    end
end
