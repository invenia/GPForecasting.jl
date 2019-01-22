# This is the most basic implementation of indep for MISO, in the sense that we are not
# playing around with kernel or anything.
# Here, the initialisation is more principled than in the other case.

function indep_exp(
    n_d::Int, # number of training days.
    its::Int, # number of gradient descent steps.
    splits::Vector{Int}, # which splits of the data to run for.
    datapath::AbstractString = "", # path for the data.
)

    # define some functions for indep experiment
    @unionise function log_pdf_indep(dist::Normal, x::AbstractArray)
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

    info("Grabbing data...")
    dat = datapath != "" ?
        mcc_training_data(n_d, "DF_train_v4.csv", datapath) :
        mcc_training_data(n_d, "DF_train_v4.csv");

    p = size(dat[1]["train_y"], 2) # Number of prices

    info("Standardising data...")
    sdata, orig = standardise_data(dat); # Normalise data

    # Add weekend column
    for s in sdata
        s["test_x"][:WKND] = [Int.(d in [1, 2, 3, 4, 5]) for d in s["test_x"][:DOW]]
        s["train_x"][:WKND] = [Int.(d in [1, 2, 3, 4, 5]) for d in s["train_x"][:DOW]]
        s["test_x"][:load] = convert(Array{Float64}, s["test_x"][:day_ahead_load_ID2])
        s["train_x"][:load] = convert(Array{Float64}, s["train_x"][:day_ahead_load_ID2])
    end

    # Temporary implementations
    function mse(means, y_true)
        return mean((y_true .- means).^2)
    end

    function mll(means, vars, y_true)
        return 0.5 * mean(log.(2π .* vars)) .+ 0.5 * mean((y_true .- means).^2 ./ vars)
    end

    # Define model
    σ² = 0.06
    Θ = [
        0.2,
        48.0,
        1e-2,
        5.0,
        0.1,
        1e-2,
        5.0,
        0.8,
        10.0,
        0.5,
        0.6,
        1.0,
        10.0
    ]
    k = NoiseKernel(
        ((Θ[1] * ((EQ() ▷ Θ[2]) * periodicise(RQ(Θ[3]) ▷ Θ[4], Fixed(24))) +
        Θ[5] * (RQ(Θ[6]) ▷ Θ[7]) +
        Θ[8] * MA(1/2) ▷ Θ[9]) ← :time) +
         (BinaryKernel(Θ[10], Θ[11]) ← :WKND) +
        (Θ[12] * EQ() ▷ Θ[13] ← :load),
        ((σ² * DiagonalKernel()) ← :time)
    )

    # Run the experiment
    out = Dict("MSE" => [], "MLL" => [], "means" => [], "vars" => [], "hyper" => [], "runtime" => [])

    info("Starting experiment...")
    for split in splits
        info("Split $split...")
        y_train = Matrix(sdata[split]["train_y"]);
        y_test = Matrix(dat[split]["test_y"]); # Let's test on untrasformed data.
        x_train = sdata[split]["train_x"];
        x_test = sdata[split]["test_x"];

        tic();
        kernel = deepcopy(k);
        gp = GP(ZeroMean(), kernel);
        gp = learn(gp, Observed(x_train), y_train, objective_indep, its=its, trace=true);

        K = gp.k(Observed(x_train))
        U = Nabla.chol(K + GPForecasting._EPSILON_ .* Eye(K))
        k_ = gp.k(Latent(x_test), Observed(x_train))
        L_y = U' \ y_train
        k_U = k_ / U

        means = k_U * L_y
        vars = repeat(diag(gp.k(Latent(x_test), Latent(x_test)) - k_U * k_U'), 1, p)

        # Un-normalise the predictions
        info("Inverse-transforming the predictions...")
        means = orig[split]["std_train"] .* means .+ orig[split]["mean_train"];
        vars = vars .* orig[split]["std_train"].^2;
        # cov = cov .* (orig[split]["std_train"].^2 * (orig[split]["std_train"].^2)');

        show(gp.k)
        println("")
        # Save results
        push!(out["MSE"], mse(means, y_test))
        push!(out["MLL"], mll(means, vars, y_test))
        push!(out["means"], means)
        push!(out["vars"], vars)
        push!(out["hyper"], exp.(gp.k[:]))
        push!(out["runtime"], toc())
        info("Split $split done!")
        println("Final results for split $split:")
        println("MSE: $(out["MSE"][end])")
        println("MLL: $(out["MLL"][end])")
    end

    info("Done!")
    return out
end

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
function indep()
    parameters = [
        [21],
        [200],
        [[31, 37, 38, 47, 49, 56, 76, 77, 80, 81] .- 14],
        [""],
            # [7, 2*7, 3*7],
            # [10, 20, 30],
            # [30],
            # [[1, 5, 25]],
            # [""],
        ]

    # -- Do not edit below this line -- #
    configuration = Dict{String, Any}(
        "parameters" => parameters,
        "experiment_function" => indep_exp,
        "seed" => 42,
        "revision" => HelloBatch.getrevinfo(GPForecasting.packagehomedir),
        "date" => now(),
    )
    return configuration
end
