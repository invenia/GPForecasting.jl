# This is the most basic implementation of the LMM for MISO, in the sense that we are not
# playing around with kernel or anything.
# Here, the initialisation is more principled than in the other case.

function describe_initialisedLMM()
    d = """
        This is a basic implementation of the LMM for MISO. It does not involve kernel
        engineering.

        Here, we initialise the parameters of each individual kernel using the projected
        latent process.
        """
    println(d)
end

source_initialisedLMM() = "initialisedLMM.jl"

function initialisedLMM_exp(
    n_d::Int, # number of training days.
    m::Int, # number of latent processes.
    its::Int, # number of gradient descent steps.
    splits::Vector{Int}, # which splits of the data to run for.
    datapath::AbstractString = "", # path for the data.
)

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
    ]
    k = (
        (
        Θ[5] * (EQ() ▷ Θ[7]) ← :DOW) +
        (BinaryKernel(Θ[10], Θ[11]) ← :WKND)
    )

    # Run the experiment
    out = Dict("MSE" => [], "MLL" => [], "means" => [], "vars" => [])

    info("Starting experiment...")
    for split in splits
        info("Split $split...")
        y_train = Matrix(sdata[split]["train_y"]);
        y_test = Matrix(dat[split]["test_y"]); # Let's test on untrasformed data.
        x_train = sdata[split]["train_x"];
        x_test = sdata[split]["test_x"];

        # Initialise the mixing matrix
        info("Initialising the mixing matrix...")
        U, S, V = svd(cov(y_train));
        H = U * diagm(sqrt.(S))[:, 1:m];

        # Find a decent initialisation for each latent process
        info("Initialising latent kernels...")
        lats = (H \ y_train')'; # project each latent process using our H
        ks = [k for i in 1:m]
        for i in 1:m
            lgp = GP(ZeroMean(), k);
            yl = @view lats[:, i]; # select a single latent process
            try # Putting this here because this step seems quite finicky
                lgp = learn(lgp, x_train, yl, objective, its=80, trace=false);
                ks[i] = set(k, lgp.k[:]);
            catch
                warn("Failed to fit latent process $i. Using default instead.")
            end
        end
        println("Initialised ks:")
        display(ks)
        info("Doing the LMM...")
        gp = GP(LMMKernel(Fixed(m), Fixed(p), Positive(σ²), Fixed(H), ks));
        try # This may fail if something blows up
            gp = learn(gp, x_train, y_train, objective, its=its, trace=false);
        catch # Ignore initilisations per latent process
            warn("Learning failed for LMM. Falling back to default kernels.")
            ks = [k for i in 1:m]
            gp = GP(LMMKernel(Fixed(m), Fixed(p), Positive(σ²), Fixed(H), ks));
            try
                gp = learn(gp, x_train, y_train, objective, its=its, trace=false);
            catch # Ignore learning entirely
                warn("Unable to learn LMM. Skipping to conditioning. Initilisations should be checked.")
            end
        end
        pos = condition(gp, x_train, y_train);
        means = pos.m(x_test);
        vars = var(pos.k, x_test);
        # cov = hourly_cov(pos.k, x_test);

        # Un-normalise the predictions
        info("Inverse-transforming the predictions...")
        means = orig[split]["std_train"] .* means .+ orig[split]["mean_train"];
        vars = vars .* orig[split]["std_train"].^2;
        # cov = cov .* (orig[split]["std_train"]' * orig[split]["std_train"]);

        println("Final ks:")
        for i in 1:length(gp.k.ks)
            show(gp.k.ks[i])
            println("")
        end
        # Save results
        push!(out["MSE"], mse(means, y_test))
        push!(out["MLL"], mll(means, vars, y_test))
        push!(out["means"], means)
        push!(out["vars"], vars)
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
function initialisedLMM()
    parameters = [
        [3 * 7],
        [20],
        [40],
        [collect(1:40)],
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
        "experiment_function" => initialisedLMM_exp,
        "seed" => 42,
        "revision" => HelloBatch.getrevinfo(GPForecasting.packagehomedir),
        "date" => now(),
    )
    return configuration
end
