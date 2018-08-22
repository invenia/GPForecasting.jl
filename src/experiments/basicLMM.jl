# This is the most basic implementation of the LMM for MISO, in the sense that we are not
# playing around with kernel or anything.
function describe_basicLMM()
    d = """
        This is a basic implementation of the LMM for MISO. It does not involve kernel
        engineering.
        """
    println(d)
end

source_basicLMM() = "basicLMM.jl"

function basicLMM_exp(
    n_d::Int, # number of training days.
    m::Int, # number of latent processes.
    its::Int, # number of gradient descent steps.
    splits::Vector{Int}, # which splits of the data to run for.
    datapath::AbstractString = "", # path for the data.
)
    dat = datapath != "" ?
        mcc_training_data(n_d, "DF_train_v4.csv", datapath) :
        mcc_training_data(n_d, "DF_train_v4.csv")

    p = size(dat[1]["train_y"], 2) # Number of prices

    sdata, orig = standardise_data(dat) # Normalise data

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
        0.35,
        1.2,
        0.35,
        1.2,
        0.27,
        0.13,
        1.5,
        0.3,
        0.11,
        2.0,
        1.0,
        1.8
    ]
    Θs = [Θ .* rand(0.5:0.05:2.0, length(Θ)) for i in 1:m]
    ks = [
        (((t[1] * periodicise(EQ() ▷ t[2], Fixed(12)) +
        t[3] * periodicise(EQ() ▷ t[4], Fixed(24)) +
        t[5] * RQ(t[6]) ▷ t[7] +
        t[8] * RQ(t[9]) ▷ t[10]) ← :time) +
        (BinaryKernel(t[11], t[12]) ← :WKND)) for t in Θs
    ]

    # Run the experiment
    out = Dict("MSE" => [], "MLL" => [], "means" => [], "vars" => [])

    for split in splits
        y_train = Matrix(sdata[split]["train_y"])
        y_test = Matrix(dat[split]["test_y"]) # Let's test on untrasformed data.
        x_train = sdata[split]["train_x"]
        x_test = sdata[split]["test_x"]

        # Initialise the mixing matrix
        U, S, V = svd(cov(y_train))
        H = U * diagm(sqrt.(S))[:, 1:m]
        gp = GP(LMMKernel(Fixed(m), Fixed(p), Positive(σ²), Fixed(H), ks));
        gp = learn(gp, x_train, y_train, objective, its=its, trace=false)
        pos = condition(gp, x_train, y_train)
        means = pos.m(x_test)
        vars = var(pos.k, x_test)

        # Un-normalise the predictions
        means = orig[split]["std_train"] .* means .+ orig[split]["mean_train"]
        vars = vars .* orig[split]["std_train"].^2

        # Save results
        push!(out["MSE"], mse(means, y_test))
        push!(out["MLL"], mll(means, vars, y_test))
        push!(out["means"], means)
        push!(out["vars"], vars)
    end

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
function basicLMM()
    parameters = [
        [7],
        [10],
        [30],
        [[1]],
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
        "experiment_function" => basicLMM_exp,
        "seed" => 42,
        "revision" => HelloBatch.getrevinfo(GPForecasting.packagehomedir),
        "date" => now(),
    )
    return configuration
end
