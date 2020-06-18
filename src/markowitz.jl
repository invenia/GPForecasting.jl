abstract type PO end

mutable struct PO_tradeoff_anal <: PO
    α::Number
end

@unionise function (po::PO_tradeoff_anal)(μ::AbstractVector, Σ::AbstractMatrix)
    A = ones(length(μ))'
    L = cholesky(Σ).L
    Z = L \ A'
    z = L \ μ
    w = 0.5 / po.α * (L' \ (L \ (μ - A'*((Z'*Z) \ (Z'*z)))))
    return w
end

mutable struct PO_maxmu_anal <: PO
    risk::Number
end

@unionise function (po::PO_maxmu_anal)(μ::AbstractVector, Σ::AbstractMatrix)
    A = ones(length(μ))'
    L = cholesky(Σ).L
    Z = L \ A'
    z = L \ μ
    λ = -(Z'*Z) \ (Z'*z)
    v = Σ \ (μ + A'*λ)
    μ = sqrt(dot(μ + A'*λ, v) / 4.0 / po.risk)
    w = 0.5 * v / μ
    return w
end

# 8.2 Total negative log likelihood of validation set
function lldeltas(
    gp::GP,
    x,
    y,
)

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

function lldeltas_obj(
    gp::GP,
    data_train::NamedTuple,
    data_val::NamedTuple,
    po::PO,
    transform::Function,
)
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

function lldeltas_obj(
    ef::EF,
    data_train::NamedTuple,
    data_val::NamedTuple,
    po::PO,
    transform::Function,
)
    return function f(params)
        nef = EF(set(ef.k, params), ef.estimator, data_train.x, data_train.y)
        obj = -lldeltas(nef::EF, data_val.x, data_val.y)
        return obj
    end
end

# 9.1 Total return of validation set
function totalreturn(
    gp::GP,
    x,
    y,
    po,
    transform,
)

    # TODO: Explore whether defining these in one call or individually is more performant
    # COMMENT: Looks like many changes would be needed for this, mostly around
    # supporting Nabla
    s = 0

    for i=1:size(x, 1)
        μ = gp.m(x[i:i, :])[:]
        Σ = Symmetric(gp.k(x[i:i, :]))
        μ, Σ = transform(μ), transform(Σ)
        w = po(μ, Σ)
        w /= norm(w)  # normalization is required to have a bounded objective
        s += dot(w, transform(y[i, :]))
    end
    return s
end

function totalreturn_obj(
    gp::GP,
    data_train::NamedTuple,
    data_val::NamedTuple,
    po::PO,
    transform::Function,
)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, data_train.x, data_train.y)
        obj = -totalreturn(pos, data_val.x, data_val.y, po, transform)
        return obj
    end
end

function totalreturn(
    ef::EF,
    x,
    y,
    po,
    transform,
)

    # TODO: Explore whether defining these in one call or individually is more performant
    # COMMENT: Looks like many changes would be needed for this, mostly around
    # supporting Nabla
    s = 0

    for i=1:size(x, 1)
        μ, Σ = ef(x[i:i, :])
        μ, Σ = transform(μ), transform(Σ)
        w = po(μ, Σ)
        w /= norm(w)  # normalization is required to have a bounded objective
        s += dot(w, transform(y[i, :]))
    end
    return s
end

function totalreturn_obj(
    ef::EF,
    data_train::NamedTuple,
    data_val::NamedTuple,
    po::PO,
    transform::Function,
)
    return function f(params)
        nef = EF(set(ef.k, params), ef.estimator, data_train.x, data_train.y)
        obj = -totalreturn(nef, data_val.x, data_val.y, po, transform)
        return obj
    end
end

# 9.3 MSE of predicted returns of validation set
function msereturns(
    gp::GP,
    x,
    y,
    w,
    po,
    transform,
)

    # TODO: Explore whether defining these in one call or individually is more performant
    # COMMENT: Looks like many changes would be needed for this, mostly around
    # supporting Nabla
    s = 0

    for i=1:size(x, 1)
        μ = gp.m(x[i:i, :])[:]
        Σ = Symmetric(gp.k(x[i:i, :]))
        μ, Σ = transform(μ), transform(Σ)
        w_pred = po(μ, Σ)
        μ_r_pred = dot(w_pred, μ)
        r_true = dot(w[i, :], transform(y[i, :])) # TODO: Calculate prior to this
        s += (μ_r_pred - r_true)^2
    end
    return s / size(x, 1)
end

function msereturns_obj(
    gp::GP,
    data_train::NamedTuple,                                                                  
    data_val::NamedTuple,                                                                    
    po::PO,                                                                                  
    transform::Function,                                                                     
)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, data_train.x, data_train.y)
        obj = msereturns(pos, data_val.x, data_val.y, data_val.w, po, transform)
        return obj
    end
end

function msereturns(
    ef::EF,
    x,
    y,
    w,
    po,
    transform,
)

    # TODO: Explore whether defining these in one call or individually is more performant
    # COMMENT: Looks like many changes would be needed for this, mostly around
    # supporting Nabla
    s = 0

    for i=1:size(x, 1)
        μ, Σ = ef(x[i:i, :])
        μ, Σ = transform(μ), transform(Σ)
        w_pred = po(μ, Σ)
        μ_r_pred = dot(w_pred, μ)
        r_true = dot(w[i, :], transform(y[i, :])) # TODO: Calculate prior to this
        s += (μ_r_pred - r_true)^2
    end
    return s / size(x, 1)
end

function msereturns_obj(
    ef::EF,
    data_train::NamedTuple,
    data_val::NamedTuple,
    po::PO,
    transform::Function,
)
    return function f(params)
        nef = EF(set(ef.k, params), ef.estimator, data_train.x, data_train.y)
        obj = msereturns(nef, data_val.x, data_val.y, data_val.w, po, transform)
        return obj
    end
end

# 9.4 Total negative log likelihood of predicted returns of validation set
function llreturns(
    gp::GP,
    x,
    y,
    w,
    po,
    transform,
)

    s = 0

    for i=1:size(x, 1)
        μ = gp.m(x[i:i, :])[:]
        Σ = Symmetric(gp.k(x[i:i, :]))
        μ, Σ = transform(μ), transform(Σ)
        w_pred = po(μ, Σ)
        # Translate the price distribution into the returns distribution through affine transformation
        μ_r_pred = dot(w_pred, μ)
        σ²_r_pred = dot(w_pred, Σ*w_pred)
        r_true = dot(w[i, :], transform(y[i, :]))
        s -= 0.5 * (log(σ²_r_pred) + (μ_r_pred - r_true)^2 / σ²_r_pred + log(2π))
    end

    return s
end

function llreturns_obj(
    gp::GP,
    data_train::NamedTuple,                                                                  
    data_val::NamedTuple,                                                                    
    po::PO,                                                                                  
    transform::Function,                                                                     
)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, data_train.x, data_train.y)
        obj = -llreturns(pos, data_val.x, data_val.y, data_val.w, po, transform)
        return obj
    end
end

function llreturns(
    ef::EF,
    x,
    y,
    w,
    po,
    transform
)

    s = 0

    for i=1:size(x, 1)
        μ, Σ = ef(x[i:i, :])
        μ, Σ = transform(μ), transform(Σ)
        w_pred = po(μ, Σ)
        # Translate the price distribution into the returns distribution through affine transformation
        μ_r_pred = dot(w_pred, μ)
        σ²_r_pred = dot(w_pred, Σ*w_pred)
        r_true = dot(w[i, :], transform(y[i, :]))
        s -= 0.5 * (log(σ²_r_pred) + (μ_r_pred - r_true)^2 / σ²_r_pred + log(2π))
    end

    return s
end

function llreturns_obj(
    ef::EF,
    data_train::NamedTuple,
    data_val::NamedTuple,
    po::PO,
    transform::Function,
)
    return function f(params)
        nef = EF(set(ef.k, params), ef.estimator, data_train.x, data_train.y)
        obj = -llreturns(nef, data_val.x, data_val.y, data_val.w, po, transform)
        return obj
    end
end

# 9.7 MSE of the weights of validation set
function msevolumes(
    gp::GP,
    x,
    y,
    w,
    po,
    transform,
)

    s = 0

    for i=1:size(x, 1)
        μ = gp.m(x[i:i, :])[:]
        Σ = Symmetric(gp.k(x[i:i, :]))
        μ, Σ = transform(μ), transform(Σ)
        w_pred = po(μ, Σ)
        s += mean((w_pred - w[i, :]).^2)
    end

    return s / size(x, 1)
end

function msevolumes_obj(
    gp::GP,
    data_train::NamedTuple,
    data_val::NamedTuple,
    po::PO,
    transform::Function,
)
    return function f(params)
        ngp = GP(gp.m, set(gp.k, params))
        pos = condition(ngp, data_train.x, data_train.y)
        obj = msevolumes(pos, data_val.x, data_val.y, data_val.w, po, transform)
        return obj
    end
end

function msevolumes(
    ef::EF,
    x,
    y,
    w,
    po,
    transform,
)

    s = 0

    for i=1:size(x, 1)
        μ, Σ = ef(x[i:i, :])
        μ, Σ = transform(μ), transform(Σ)
        w_pred = po(μ, Σ)
        s += mean((w_pred - w[i, :]).^2)
    end

    return s / size(x, 1)
end

function msevolumes_obj(
    ef::EF,
    data_train::NamedTuple,
    data_val::NamedTuple,
    po::PO,
    transform::Function,
)
    return function f(params)
        nef = EF(set(ef.k, params), ef.estimator, data_train.x, data_train.y)
        obj = msevolumes(nef, data_val.x, data_val.y, data_val.w, po, transform)
        return obj
    end
end
