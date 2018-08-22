# This is the most basic implementation of the OLMM for MISO. Just shows how it works.
function describe_basicOLMM()
    d = """
        This is the most basic implementation of the OLMM for MISO. Just shows how it works.
        No learning is done.
        """
    println(d)
end

source_basicOLMM() = "basicOLMM.jl"

function basicOLMM_exp(
    n_d::Int, # number of training days.
    m::Int, # number of latent processes.
    σ²::Float64, # observation noise (note that we are working on the standardised space)
    lat_noise::Float64, # noise for the latent processes
    splits::Vector{Int}, # which splits of the data to run for.
    datapath::AbstractString = "", # path for the data.
)
    dat = datapath != "" ?
        mcc_training_data(n_d, "DF_train_v4.csv", datapath) :
        mcc_training_data(n_d, "DF_train_v4.csv");

    p = size(dat[1]["train_y"], 2) # Number of prices

    sdata, orig = standardise_data(dat); # Normalise data

    # Define model
    k = (
        ((0.15 * (MA(1/2) ▷ 24.0) + 0.15 * (EQ() ▷ 36.0)) ← :time) +
        0.7 * ((EQ() ▷ 5.0) ← :day_ahead_load_ID2) *
        (periodicise((EQ() ▷ 0.5), Fixed(24)) * (EQ() ▷ (24.0 * 4.0)) ← :time)
    )

    k = (
        (0.5 * stretch(EQ(), 24.0*3.0) * periodicise(EQ() ▷ 1.0, Fixed(24.0)) ← :time) +
        (0.5 * ((EQ() ▷ 5.0) ← :day_ahead_load_ID2))
    )

    # Run the experiment
    out = Dict("MSEs" => [], "MLLs" => [], "means" => [])

    for split in splits
        y_train = disallowmissing(Matrix(sdata[split]["train_y"]))
        y_test = disallowmissing(Matrix(dat[split]["test_y"])) # Let's test on untrasformed data.
        x_train = sdata[split]["train_x"]
        x_test = sdata[split]["test_x"]

        # Initialise the mixing matrix
        U, S, V = svd(cov_LW(y_train))
        H = U * diagm(sqrt.(S))[:, 1:m]
        gp = GP(OLMMKernel(m, p, σ², lat_noise, H, k));
        gp = learn(gp, x_train, y_train, objective, its=30)
        pos = condition(gp, x_train, y_train)
        means = pos.m(x_test)
        covs = hourly_cov(pos.k, x_test)

        # Un-normalise the predictions
        means = orig[split]["std_train"] .* means .+ orig[split]["mean_train"]
        stds = orig[split]["std_train"]' * orig[split]["std_train"]
        covs = BlockDiagonal([Hermitian(b .* stds) for b in blocks(covs)])

        mvns = [MvNormal(means[i, :], Matrix(blocks(covs)[i])) for i in 1:24]

        # Save results
        push!(out["MSEs"], [mse(mvns[i], y_test[i, :]) for i in 1:24])
        push!(out["MLLs"], [-logpdf(mvns[i], y_test[i, :]) for i in 1:24])
        # push!(out["means"], means)
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
function basicOLMM()
    parameters = [
        [3*7],
        [100],
        [0.1],
        [5.0],
        [[1]],
        [""],
    ]

    # -- Do not edit below this line -- #
    configuration = Dict{String, Any}(
        "parameters" => parameters,
        "experiment_function" => basicOLMM_exp,
        "seed" => 42,
        "revision" => HelloBatch.getrevinfo(GPForecasting.packagehomedir),
        "date" => now(),
    )
    return configuration
end
