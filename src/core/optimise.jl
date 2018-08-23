export minimise, learn

"""
    minimise(f::Function,
        x_init::Vector;
        its=200,
        trace=true,
        alphaguess=LineSearches.InitialStatic(scaled=true),
        linesearch=LineSearches.BackTracking(),
    ) -> Vector

Minimise objective funcion `f`, starting with initial configuration `x_init`, for a miaximum
`its` iterations. If `trace`, runs verbose version. `alphaguess` and `linesearch` are the
initial and optimisation linesearches. Returns the optimised parameters. `f`
must be a function of `x` only.
"""

function minimise(
    f::Function,
    x_init::Vector;
    its=200,
    trace=true,
    alphaguess=LineSearches.InitialStatic(scaled=true),
    linesearch=LineSearches.BackTracking(),
)

    ∇grad   = ∇(f)
    function grad!(storage::Vector, x::Vector)
        storage[:] = ∇grad(x)[1]
    end
    return optimize(
        f,
        grad!,
        x_init,
        LBFGS(alphaguess = alphaguess, linesearch = linesearch),
        Optim.Options(
            x_tol = 1.0e-8,
            f_tol = 1.0e-8,
            g_tol = 1.0e-5,
            f_calls_limit = its,
            g_calls_limit = its,
            iterations = its,
            show_trace = trace,
        ),
    ).minimizer

end

"""
    learn(gp::GP,
        x,
        y,
        obj::Function;
        Θ_init::Array=[],
        its=200,
        trace=true,
        alphaguess=LineSearches.InitialStatic(scaled=true),
        linesearch=LineSearches.BackTracking(),
    ) -> GP

Obtain the parameters that minimise the `obj` of a `gp` over points `x` with observation
values `y`. `obj` can be any function of `gp`, `x`, `y` and `Θ_init` (only). `Θ_init`
determines the starting point. `its` is the miaximum number of iterations. If `trace`,
runs verbose version. `alphaguess` and `linesearch` are the
initial and optimisation linesearches. Returns a `GP` with the optimised parameters.
"""
function learn(
    gp::GP,
    x,
    y,
    obj::Function;
    Θ_init::Array=[],
    its=200,
    trace=true,
    alphaguess=LineSearches.InitialStatic(scaled=true),
    linesearch=LineSearches.BackTracking(),
)
    Θ_init = isempty(Θ_init) ? gp.k[:] : Θ_init
    Θ_opt = minimise(
        obj(gp, x, y),
        Θ_init,
        its=its,
        trace=trace,
        alphaguess=alphaguess,
        linesearch=linesearch,
    )
    return GP(gp.m, set(gp.k, Θ_opt)) # Again, assuming we are only optimising kernels
    # Got to overload if we want parameters in the means as well
end

#NOTE: The code below currently breaks due to Nabla issues. TODO: Fix it.
function learn(
    gp::GP{OLMMKernel, <:Mean},
    x,
    y,
    obj::Function;
    opt_U=false,
    K_U_cycles::Int=0,
    Θ_init::Array=[],
    its=200,
    trace=true,
    alphaguess=LineSearches.InitialStatic(scaled=true),
    linesearch=LineSearches.BackTracking(),
)
    ngp = deepcopy(gp)
    if opt_U
        U = greedy_U(gp.k, x, y)
        H, P = build_H_and_P(U, unwrap(gp.k.S_sqrt))
        ngp.k.U, ngp.k.H, ngp.k.P = Fixed(U), Fixed(H), Fixed(P)
    end
    if K_U_cycles == 0
        Θ_init = isempty(Θ_init) ? ngp.k[:] : Θ_init
        Θ_opt = minimise(
            obj(ngp, x, y),
            Θ_init,
            its=its,
            trace=trace,
            alphaguess=alphaguess,
            linesearch=linesearch,
        )
        return GP(ngp.m, set(ngp.k, Θ_opt)) # Again, assuming we are only optimising kernels
    # Got to overload if we want parameters in the means as well
    end
    for i in 1:K_U_cycles
        Θ_init = isempty(Θ_init) ? ngp.k[:] : Θ_init
        Θ_opt = minimise(
            obj(ngp, x, y),
            Θ_init,
            its=its,
            trace=trace,
            alphaguess=alphaguess,
            linesearch=linesearch,
        )
        ngp.k = set(ngp.k, Θ_opt)
        U = greedy_U(gp.k, x, y)
        H, P = build_H_and_P(U, unwrap(gp.k.S_sqrt))
        ngp.k.U, ngp.k.H, ngp.k.P = Fixed(U), Fixed(H), Fixed(P)
    end
    return ngp
end