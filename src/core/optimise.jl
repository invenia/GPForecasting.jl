"""
    minimise(f::Function,
        x_init::Vector;
        its=200,
        trace=true,
        algorithm::Type{<:Optim.FirstOrderOptimizer}=LBFGS,
        alphaguess=LineSearches.InitialStatic(scaled=true),
        linesearch=LineSearches.BackTracking(),
        kwargs...
    ) -> Vector

Minimise objective funcion `f`, starting with initial configuration `x_init`, for a miaximum
`its` iterations. If `trace`, runs verbose version. `algorithm` is a first order optimization
method while `alphaguess` and `linesearch` are the initial and optimisation linesearches.
Returns the optimised parameters. `f` must be a function of `x` only. `kwargs...` are additional
keyword arguments for `algorithm`.
"""
function minimise(
    f::Function,
    x_init::Vector;
    its=200,
    trace=true,
    algorithm::Type{<:Optim.FirstOrderOptimizer}=LBFGS,
    alphaguess=LineSearches.InitialStatic(scaled=true),
    linesearch=LineSearches.BackTracking(),
    kwargs...
)

    ∇grad   = ∇(f)
    function grad!(storage::Vector, x::Vector)
        storage[:] = ∇grad(x)[1]
    end
    res = optimize(
        f,
        grad!,
        x_init,
        algorithm(alphaguess = alphaguess, linesearch = linesearch, kwargs...),
        Optim.Options(
            x_tol = 1.0e-8,
            f_tol = 1.0e-8,
            g_tol = 1.0e-5,
            f_calls_limit = its,
            g_calls_limit = its,
            iterations = its,
            show_trace = trace,
        ),
    )

    return res.minimizer
end

"""
    minimise_summary(f::Function,
        x_init::Vector;
        its=200,
        trace=true,
        algorithm::Type{<:Optim.FirstOrderOptimizer}=LBFGS,
        alphaguess=LineSearches.InitialStatic(scaled=true),
        linesearch=LineSearches.BackTracking(),
        kwargs...
    ) -> Optim.jl Optimization Object

Like `minmise()`, but returns a summary of the optimization results.
"""
function minimise_summary(
    f::Function,
    x_init::Vector;
    its=200,
    trace=true,
    algorithm::Type{<:Optim.FirstOrderOptimizer}=LBFGS,
    alphaguess=LineSearches.InitialStatic(scaled=true),
    linesearch=LineSearches.BackTracking(),
    kwargs...
)

    ∇grad   = ∇(f)
    function grad!(storage::Vector, x::Vector)
        storage[:] = ∇grad(x)[1]
    end
    res = optimize(
        f,
        grad!,
        x_init,
        algorithm(alphaguess = alphaguess, linesearch = linesearch, kwargs...),
        Optim.Options(
            x_tol = 1.0e-8,
            f_tol = 1.0e-8,
            g_tol = 1.0e-5,
            f_calls_limit = its,
            g_calls_limit = its,
            iterations = its,
            show_trace = trace,
        ),
    )

    return res
end

"""
    learn(gp::GP,
        x,
        y,
        obj::Function;
        Θ_init::Array=[],
        its=200,
        trace=true,
        algorithm::Type{<:Optim.FirstOrderOptimizer}=LBFGS,
        alphaguess=LineSearches.InitialStatic(scaled=true),
        linesearch=LineSearches.BackTracking(),
        kwargs...
    ) -> GP

Obtain the parameters that minimise the `obj` of a `gp` over points `x` with observation
values `y`. `obj` can be any function of `gp`, `x`, `y` and `Θ_init` (only). `Θ_init`
determines the starting point. `its` is the miaximum number of iterations. If `trace`,
runs verbose version. `algorithm` is a first order optimization method, while `alphaguess`
and `linesearch` are the initial and optimisation linesearches.
Returns a `GP` with the optimised parameters. `kwargs...` are additional keyword arguments
for `algorithm`.
"""
function learn(
    gp::GP,
    x,
    y::AbstractArray{<:Real},
    obj::Function;
    Θ_init::Array=[],
    its=200,
    trace=true,
    algorithm::Type{<:Optim.FirstOrderOptimizer}=LBFGS,
    alphaguess=LineSearches.InitialStatic(scaled=true),
    linesearch=LineSearches.BackTracking(),
    kwargs...
)
    Θ_init = isempty(Θ_init) ? gp.k[:] : Θ_init
    Θ_opt = minimise(
        obj(gp, x, y),
        Θ_init,
        its=its,
        trace=trace,
        algorithm=algorithm,
        alphaguess=alphaguess,
        linesearch=linesearch,
        kwargs...
    )

    # Again, assuming we are only optimising kernels
    # Got to overload if we want parameters in the means as well
    return GP(gp.m, set(gp.k, Θ_opt))
end

"""
    learn_summary(gp::GP,
        x,
        y,
        obj::Function;
        Θ_init::Array=[],
        its=200,
        trace=true,
        algorithm::Type{<:Optim.FirstOrderOptimizer}=LBFGS,
        alphaguess=LineSearches.InitialStatic(scaled=true),
        linesearch=LineSearches.BackTracking(),
        kwargs...
    ) -> Optim.jl Optimization Object, GP

Like `learn()`, but returns a tuple with the summary of the optimization procedure and the learned GP.
"""
function learn_summary(
    gp::GP,
    x,
    y::AbstractArray{<:Real},
    obj::Function;
    Θ_init::Array=[],
    its=200,
    trace=true,
    algorithm::Type{<:Optim.FirstOrderOptimizer}=LBFGS,
    alphaguess=LineSearches.InitialStatic(scaled=true),
    linesearch=LineSearches.BackTracking(),
    kwargs...
)
    Θ_init = isempty(Θ_init) ? gp.k[:] : Θ_init
    Θ_opt = minimise_summary(
        obj(gp, x, y),
        Θ_init,
        its=its,
        trace=trace,
        algorithm=algorithm,
        alphaguess=alphaguess,
        linesearch=linesearch,
        kwargs...
    )

    # Again, assuming we are only optimising kernels
    # Got to overload if we want parameters in the means as well
    return Θ_opt, GP(gp.m, set(gp.k, Θ_opt.minimizer))
end

function learn(
    gp::GP{OLMMKernel, <:Mean},
    x,
    y::AbstractMatrix{<:Real},
    obj::Function;
    opt_U=false,
    K_U_cycles::Int=0,
    Θ_init::Array=[],
    its=200,
    trace=true,
    algorithm::Type{<:Optim.FirstOrderOptimizer}=LBFGS,
    alphaguess=LineSearches.InitialStatic(scaled=true),
    linesearch=LineSearches.BackTracking(),
    kwargs...
)
    ngp = deepcopy(gp)
    if opt_U
        U = greedy_U(gp.k, x, y)
        H, P = build_H_and_P(U, unwrap(gp.k.S_sqrt))
        ngp.k.U, ngp.k.H, ngp.k.P = Fixed(U), Fixed(H), Fixed(P)
    end
    Θ_init = isempty(Θ_init) ? ngp.k[:] : Θ_init
    if K_U_cycles == 0
        Θ_opt = minimise(
            obj(ngp, x, y),
            Θ_init,
            its=its,
            trace=trace,
            algorithm=algorithm,
            alphaguess=alphaguess,
            linesearch=linesearch,
            kwargs...
        )
        return GP(ngp.m, set(ngp.k, Θ_opt)) # Again, assuming we are only optimising kernels
    # Got to overload if we want parameters in the means as well
    end
    for i in 1:K_U_cycles
        Θ_opt = minimise(
            obj(ngp, x, y),
            Θ_init,
            its=its,
            trace=trace,
            algorithm=algorithm,
            alphaguess=alphaguess,
            linesearch=linesearch,
            kwargs...
        )
        ngp.k = set(ngp.k, Θ_opt)
        U = greedy_U(ngp.k, x, y)
        H, P = build_H_and_P(U, unwrap(ngp.k.S_sqrt))
        ngp.k.U, ngp.k.H, ngp.k.P = Fixed(U), Fixed(H), Fixed(P)
        Θ_init = Θ_opt
    end
    return ngp
end

"""
    learn_sparse(
        gp::GP,
        x,
        y::AbstractArray{<:Real},
        Xm,
        σ²,
        obj::Function=titsiasobj;
        Θ_init::Array=[],
        its=200,
        trace=true,
        alphaguess=LineSearches.InitialStatic(scaled=true),
        linesearch=LineSearches.BackTracking(),
    )

Learn a GP using some sparse approximation. By default, Titsias' is used. Note that this
function expects a regular `GP` as input. The sparsification is performed under the hood.
`x` represents the training input locations, `y` the training outputs, `Xm` the inducing
point locations, `σ²` the noise and `obj` controls which type of approximation to use.
The outputs are the optimised `GP` (of the same type as the input one), the optimised
inducing point locations `Xm` and the optimised noise `σ²` (also with the same type as
the inputs).
"""
function learn_sparse(
    gp::GP,
    x,
    y::AbstractArray{<:Real},
    Xm,
    σ²,
    obj::Function=titsiasobj;
    Θ_init::Array=[],
    its=200,
    trace=true,
    alphaguess=LineSearches.InitialStatic(scaled=true),
    linesearch=LineSearches.BackTracking(),
)

    Θ_init = isempty(Θ_init) ? vcat(pack(Xm), pack(σ²), gp.k[:]) : Θ_init
    Θ_opt = minimise(
        obj(gp, x, y, Xm, σ²),
        Θ_init,
        its=its,
        trace=trace,
        alphaguess=alphaguess,
        linesearch=linesearch,
    )

    # Again, assuming we are only optimising kernels
    # Got to overload if we want parameters in the means as well
    sk = SparseKernel(gp.k, Xm, Fixed(size(unwrap(Xm), 1)), σ²)
    sparse_gp = GP(gp.m, set(sk, Θ_opt))
    return GP(sparse_gp.m, sparse_gp.k.k), sparse_gp.k.Xm, sparse_gp.k.σ²
end
