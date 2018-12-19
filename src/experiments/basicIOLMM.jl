"""
    set_parameters()

This function sets the parameters of your experiment.
The first dictionary, `parameters`, should be populated with `string => AbstractArray`
key-value pairs, where each `AbstractArray` contains the parameter values you would wish to
iterate over. The array could also only contain one element if you would like to use one
constant parameter.
The second dictionary should not be altered, it is used in the experiment execution. This
dictionary references the `experiment_function`, which should contain your experiment.
"""
function basicIOLMM()
    parameters = [
        [3],   # number of weeks
        [300], # number of latent processes
        [0.1], # observation noise (std)
        [5.0], # latent noise
        [25],  # number of iterations
        [[133]],   # number of splits
        ["DF_train_deltas_90.csv"],
        ["s3://invenia-research-datasets/ProbabilisticForecasting/MISO/v1"],
        ]

        # -- Do not edit below this line -- #
    configuration = Dict{String, Any}(
        "parameters" => parameters,
        "experiment_function" => basicIOLMM_exp,
        "seed" => 42,
        "revision" => HelloBatch.getrevinfo(GPForecasting.packagehomedir),
        "date" => now(),
    )
    return configuration
end

# This is the most basic implementation of the IOLMM for MISO. Just shows how it works.

function describe(x::typeof(basicIOLMM))
    d = """
        This is the most basic implementation of the IOLMM for MISO. Just shows how it works.
        """
    return d
end

source(x::typeof(basicIOLMM)) = "basicIOLMM.jl"

function basicIOLMM_exp(
    n_w::Int, # number of training weeks.
    m::Int, # number of latent processes.
    obs_noise::Float64, # observation noise (note that we are working on the standardised space)
    lat_noise::Float64, # latent noise
    its::Int, # number of gradient descent steps.
    splits::Vector{Int}, # which splits of the data to run for.
    datafile::AbstractString = "", # filename of data
    datapath::AbstractString = "", # path for the data.
)
    # define some functions for use in experiment
    function mse(means, y_true)
        return mean((y_true .- means).^2)
    end

    @unionise function log_pdf_indep(dist::Gaussian, x::AbstractArray)
        U = Nabla.chol(dist)
        z = U' \ (x .- dist.μ)
        log_det = 2.0 * size(x, 2) * sum(log.(diag(U)))
        return -0.5 * (log_det + prod(size(x)) * log(2π) + sum(z .* z))
    end

    @unionise function log_pdf_indep(
        gp::GP,
        x,
        y::AbstractArray,
        params::Vector{G}
        ) where {G <: Real}
            ngp = GP(gp.m, set(gp.k, params))
        return log_pdf_indep(ngp::GP, x, y)
    end

    @unionise function log_pdf_indep(gp::GP, x, y::AbstractArray)
        return log_pdf_indep(gp(x), y)
    end

    @unionise function objective_indep(gp::GP, x, y::AbstractArray)
        return function f(params)
            return -log_pdf_indep(gp::GP, x, y, params)
        end
    end

    info("LAM: n_w = ", n_w, " m = ", m, " group = ", group)

    info("Grabbing data...")

    profile = haskey(ENV, "AWS_BATCH_JOB_ID") ? nothing : "ProbabilisticForecasting@Invenia"
    fpath = HelloBatch.get_s3_object(datafile, datapath, profile=profile, localdir=GPForecasting.packagehomedir, overwrite=false)

    data = CSV.read(joinpath(GPForecasting.packagehomedir, datafile))

    # Define model
    # terms of time kernel
    k_time_1 = (1.0 * periodicise(MA(1/2) ▷  Fixed(0.1), Fixed(24.0)) * (RQ(2.0) ▷ Fixed(24.0 * 3.5)))
    k_time_2 = (1.0 * periodicise(MA(1/2) ▷  Fixed(0.5), Fixed(24.0)) * (RQ(2.0) ▷ Fixed(24.0 * 3.5)))
    k_time_3 = (1.0 * periodicise(MA(1/2) ▷  Fixed(0.1), Fixed(24.0)) * (RQ(2.0) ▷ Fixed(24.0 * 7.0)))
    k_time_4 = (1.0 * periodicise(MA(1/2) ▷  Fixed(0.5), Fixed(24.0)) * (RQ(2.0) ▷ Fixed(24.0 * 7.0)))
    k_time_5 = (1.0 * periodicise(MA(1/2) ▷  Fixed(0.1), Fixed(24.0)) * (RQ(2.0) ▷ Fixed(24.0 * 14.0)))
    k_time_6 = (1.0 * periodicise(MA(1/2) ▷  Fixed(0.5), Fixed(24.0)) * (RQ(2.0) ▷ Fixed(24.0 * 14.0)))

    # composite time kernel
    k_time = k_time_1 +
             k_time_2 +
             k_time_3 +
             k_time_4 +
             k_time_5 +
             k_time_6

    # load kernel
    k_load = (EQ() ▷ 5.0)

    # composite kernel
    k = NoiseKernel(
        ( k_time ← :time ) * ( k_load ← :day_ahead_load_ID2 ),
        lat_noise * DiagonalKernel() ← :time,
       )

    # Run the experiment
    # save results before (pre) and after (opt) optimisation
    out = Dict(
            "MSE_pre" => [], "MLL_COV_pre" => [], "MSEs_pre" => [],
            "MLL_COVs_pre" => [], "means_pre" => [], "hyper_pre" => [],
            "MSE_opt" => [], "MLL_COV_opt" => [], "MSEs_opt" => [],
            "MLL_COVs_opt" => [], "means_opt" => [], "hyper_opt" => [],
            "runtime" => [],
              )

    info("Starting experiment...")
    for split in splits
        tic();
        info("Split $split...")
        # get training and test outputs
        y_train = convert.(Float64, Matrix(data[(split-n_w*7-2)*24+1:(split-2)*24, 8:end]))
        y_test = convert.(Float64, Matrix(data[(split-1)*24+1:split*24, 8:end]))
        # get training and test inputs
        x_train = data[(split-n_w*7-2)*24+1:(split-2)*24, 1:7]
        x_test = data[(split-1)*24+1:split*24,1:7]

        # add time feature variable
        x_train[:time] = collect(1:n_w*7*24)
        x_test[:time] = collect(n_w*7*24+25:n_w*7*24+48)

        # basic statistics of training output required later
        y_train_mean = meandims(y_train, 1)
        y_train_std = stddims(y_train, 1)
        y_train_var = y_train_std.^2

        p = size(y_train, 2)

        # compute standardised training output
        y_train_standardised = (y_train .- y_train_mean) ./ y_train_std

        # initialise the mixing matrix
        info("Initialising the mixing matrix...")
        U, S, V = svd(cov_LW(y_train_standardised))
        H = U * diagm(sqrt.(S))[:, 1:m]

        # compute transfromed training output
        y_train_transformed = (H \ y_train_standardised')'

        info("Doing the OLMM...")

        # training without hyperparameter optimisation
        # (identical kernels for all latent processes)
        info("=> Pre")

        kernel = deepcopy(k)
        gp = GP(ZeroMean(), kernel)

        # this below would be a hyperparameter optimisation
        # of a single fixed kernel for all latent processes
        # gp = learn(gp, Observed(x_train), y_train_transformed, objective_indep, its=its, trace=true)

        K = gp.k(Observed(x_train))
        U = Nabla.chol(K + GPForecasting._EPSILON_ .* Eye(K))
        k_ = gp.k(Latent(x_test), Latent(x_train))
        L_y = U' \ y_train_transformed
        k_U = k_ / U

        # prediction of mean and variances of the independent latent processes
        means_ = k_U * L_y
        vars_ = repmat(diag(gp.k(Observed(x_test)) - k_U * k_U'), 1, m)

        # transform latent processes back to original space
        means = (H * means_')' .* y_train_std .+ y_train_mean
        covs = []
        for i = 1:24
            push!(covs, Matrix(Hermitian(y_train_std .* (H * diagm(vars_[i,:]) * H' + obs_noise * I) .* y_train_std')))
        end

        # set MvNormals
        mvns = [GPForecasting.MvNormal(means[i, :], covs[i]) for i in 1:24]

        # compute hourly score functions
        mses = [mse(mvns[i].μ, y_test[i, :]) for i in 1:24]
        mll_covs = [-logpdf(mvns[i], y_test[i, :]) for i in 1:24]

        println("")
        # save results
        push!(out["MSEs_pre"], mses)
        push!(out["MLL_COVs_pre"], mll_covs)
        push!(out["MSE_pre"], mean(mses))
        push!(out["MLL_COV_pre"], mean(mll_covs))
        push!(out["means_pre"], means)
        push!(out["hyper_pre"], exp.(gp.k[:]))
        println("Pre results for split $split:")
        println("MSE_pre: $(out["MSE_pre"][end])")
        println("MLL_COV_pre: $(out["MLL_COV_pre"][end])")

        # training with hyperparameter optimisation
        # (independent optimisations for all latent processes))
        info("=> Opt")

        means_ = zeros(24, m)
        vars_ = zeros(24, m)
        hypers = []
        for i = 1:m
            kernel = deepcopy(k)
            gp = GP(ZeroMean(), kernel)
            # independent hyperparameter optimisations
            gp = learn(gp, Observed(x_train), y_train_transformed[:,i], objective, its=its, trace=false)
            pos = condition(gp, Observed(x_train), y_train_transformed[:,i])
            # prediction of mean and variances of the independent latent processes
            means_[:,i] = pos.m(Observed(x_test))
            vars_[:,i] = diag(pos.k(Observed(x_test)))
            append!(hypers, exp.(gp.k[:]))
        end

        # transform latent processes back to original space
        means = (H * means_')' .* y_train_std .+ y_train_mean
        covs = []
        for i = 1:24
            push!(covs, Matrix(Hermitian(y_train_std .* (H * diagm(vars_[i,:]) * H' + obs_noise * I) .* y_train_std')))
        end

        # set MvNormals
        mvns = [GPForecasting.MvNormal(means[i, :], covs[i]) for i in 1:24]

        # compute hourly score functions
        mses = [mse(mvns[i].μ, y_test[i, :]) for i in 1:24]
        mll_covs = [-logpdf(mvns[i], y_test[i, :]) for i in 1:24]

        println("")
        # save results
        push!(out["MSEs_opt"], mses)
        push!(out["MLL_COVs_opt"], mll_covs)
        push!(out["MSE_opt"], mean(mses))
        push!(out["MLL_COV_opt"], mean(mll_covs))
        push!(out["means_opt"], means)
        push!(out["hyper_opt"], hypers)
        println("Opt results for split $split:")
        println("MSE_opt: $(out["MSE_opt"][end])")
        println("MLL_COV_opt: $(out["MLL_COV_opt"][end])")

        push!(out["runtime"], toc())
        info("Split $split done!")
    end

    info("Done!")
    return out
end
