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
function hour_loadLMM()
    parameters = [
        [7], #[14], #[21],
        [10], #[15], #[20],
        [50],
        [[31, 37, 38, 47, 49, 56, 76, 77, 80, 81]],
        #[[24, 30, 31, 40, 42, 49, 69, 70, 73, 74]], # for 2 weeks training data
        #[[17, 23, 24, 33, 35, 42, 62, 63, 66, 67]], # for 3 weeks training data
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
        "experiment_function" => hour_loadLMM_exp,
        "seed" => 42,
        "revision" => HelloBatch.getrevinfo(GPForecasting.packagehomedir),
        "date" => now(),
    )
    return configuration
end

# This is the most basic implementation of the LMM for MISO, in the sense that we are not
# playing around with kernel or anything.
# Here, the initialisation is more principled than in the other case.
function describe(x::typeof(hour_loadLMM))
    d = """
        This is a basic implementation of the LMM for MISO. It does not involve kernel
        engineering.
        """
    return d
end

source(x::typeof(hour_loadLMM)) = "hour_loadLMM.jl"

function hour_loadLMM_exp(
    n_d::Int, # number of training days.
    m::Int, # number of latent processes.
    its::Int, # number of gradient descent steps.
    splits::Vector{Int}, # which splits of the data to run for.
    datapath::AbstractString = "", # path for the data.
)

    info("n_d = ", n_d, " m = ", m)

    info("Grabbing data...")
    dat = datapath != "" ?
        mcc_training_data(n_d, "DF_train_v4.csv", datapath) :
        mcc_training_data(n_d, "DF_train_v4.csv");

    p = size(dat[1]["train_y"], 2) # Number of prices

    info("Standardising data...")
    sdata, orig = standardise_data(dat); # Normalise data

    # Add weekend column
    for s in sdata
        # weekend is omitted
#       s["test_x"][:WKND] = [Int.(d in [1, 2, 3, 4, 5]) for d in s["test_x"][:DOW]]
#       s["train_x"][:WKND] = [Int.(d in [1, 2, 3, 4, 5]) for d in s["train_x"][:DOW]]
        # total load added as new column
        s["test_x"][:load] = convert(Array{Float64}, s["test_x"][:day_ahead_load_ID2])
        s["train_x"][:load] = convert(Array{Float64}, s["train_x"][:day_ahead_load_ID2])
    end

    # Temporary implementations
    function hcov(k::MultiOutputKernel, x)
        ks = [k(x[i,:]) for i=1:size(x)[1]] # we need this for DataFrames
        return hcov(k, x, ks)
    end

    function hcov(k::MultiOutputKernel, x, ks::Vector)
        n = size(x, 1)
        p = size(ks[1], 1)
        res = zeros(n, p, p)
        for i in 1:length(ks)
            res[i, :, :] = ks[i]
        end
        return res
    end

    function mse(means, y_true)
        return mean((y_true .- means).^2)
    end

    function mll_var(means, vars, y_true)
        return 0.5 * mean(log.(2π .* vars)) .+ 0.5 * mean((y_true .- means).^2 ./ vars)
    end

    function mll_cov(means, covs, y_true)
        h = size(means)[1]
        n = size(means)[2]
        mll = 0.0
        for i=1:h
            C = covs[i,:,:]
            for j=1:n
                C[j,j] += GPForecasting._EPSILON_
            end
            L = Nabla.chol(Symmetric(C))'
            z = L \ (y_true[i,:] .- means[i,:])
            mll += 0.5 * (n * log(2π) + 2.0 * sum(log.(diag(L))) + dot(z,z))
        end
        return mll / h
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
    k = (
        # kernel for hours
        ((Θ[1] * ((EQ() ▷ Θ[2]) * periodicise(RQ(Θ[3]) ▷ Θ[4], Fixed(24))) +
        Θ[5] * (RQ(Θ[6]) ▷ Θ[7]) +
        Θ[8] * MA(1/2) ▷ Θ[9]) ← :time) +
        # weekend kernel is omitted
#       (BinaryKernel(Θ[10], Θ[11]) ← :WKND) +
        # kernel for total load
        (Θ[12] * EQ() ▷ Θ[13] ← :load)
    )

    # Run the experiment
    out = Dict("MSE" => [], "MLL" => [], "MLL_COV" => [], "means" => [], "vars" => [], "covs" => [], "hyper" => [], "runtime" => [])

    info("Starting experiment...")
    for split in splits
        tic();
        info("Split $split...")
        y_train = Matrix(sdata[split]["train_y"]);
        y_test = Matrix(dat[split]["test_y"]); # Let's test on untrasformed data.
        x_train = sdata[split]["train_x"];
        x_test = sdata[split]["test_x"];

        # Initialise the mixing matrix
        info("Initialising the mixing matrix...")
        U, S, V = svd(cov(y_train));
        H = U * Diagonal(sqrt.(S))[:, 1:m];

        # Find a decent initialisation for each latent process
        info("Initialising latent kernels...")
        lats = (H \ y_train')'; # project each latent process using our H
        ks = [k for i in 1:m]
        ts = []
        for i in 1:m
            tic()
            lgp = GP(ZeroMean(), k);
            yl = @view lats[:, i]; # select a single latent process
            try # Putting this here because this step seems quite finicky
                lgp = learn(lgp, x_train, yl, objective, its=100, trace=false);
                ks[i] = set(k, lgp.k[:]);
            catch
                warn("Failed to fit latent process $i. Using default instead.")
            end
            push!(ts, toc())
        end
        println("Initialised ks:")
        display(ks)
        info("Doing the LMM...")
        gp = GP(LMMKernel(Fixed(m), Fixed(p), Positive(σ²), Fixed(H), ks));
        tic()
        try # This may fail if something blows up
            gp = learn(gp, x_train, y_train, objective, its=its, trace=false);
        catch # Ignore initilisations per latent process
            warn("Learning failed for LMM. Falling back to default kernels.")
            ks = [k for i in 1:m]
            gp = GP(LMMKernel(Fixed(m), Fixed(p), Positive(σ²), Fixed(H), ks));
            try
                gp = learn(gp, x_train, y_train, objective, its=its, trace=true);
            catch # Ignore learning entirely
                warn("Unable to learn LMM. Skipping to conditioning. Initilisations should be checked.")
            end
        end
        push!(ts, toc())
        pos = condition(gp, x_train, y_train);
        means = pos.m(x_test);
        vars = var(pos.k, x_test);
        covs = hcov(pos.k, x_test);

        # Un-normalise the predictions
        info("Inverse-transforming the predictions...")
        means = orig[split]["std_train"] .* means .+ orig[split]["mean_train"];
        vars = vars .* orig[split]["std_train"].^2;
        for i=1:size(covs)[1]
            covs[i,:,:] = orig[split]["std_train"] .* covs[i,:,:] .* orig[split]["std_train"]';
        end

        println("Final ks:")
        for i in 1:length(gp.k.ks)
            show(gp.k.ks[i])
            println("")
        end
        # Save results
        push!(out["MSE"], mse(means, y_test))
        push!(out["MLL"], mll_var(means, vars, y_test))
        push!(out["MLL_COV"], mll_cov(means, covs, y_test))
        push!(out["means"], means)
        push!(out["vars"], vars)
        #push!(out["covs"], covs)
        push!(out["hyper"], exp.(gp.k[:]))
        push!(out["runtime"], [ts, toc()])
        info("Split $split done!")
        println("Final results for split $split:")
        println("MSE: $(out["MSE"][end])")
        println("MLL: $(out["MLL"][end])")
        println("MLL_COV: $(out["MLL_COV"][end])")
    end

    info("Done!")
    return out
end
