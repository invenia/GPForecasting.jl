function describe_damec()
    d = """
        This is a basic implementation of GPs for forecasting DAMEC in MISO. It involves kernel
        engineering.
        """
    println(d)
end

source_damec() = "damec.jl"

@everywhere log_transform(x) = sign.(x).*log.(abs.(x) .+ 1)
@everywhere inv_log_transform(x) = sign.(x).*(exp.(abs.(x)) .- 1)

# Function computing weights for the exponential model
@everywhere function w(t::Union{AbstractArray}, n_w::Integer)
    λ = 1-exp(-4/(7*n_w))
    weights = λ.*(1 .- λ).^(1.-t)
    return weights / sum(weights)
end

# Calculates the weighted mean from data and weights, used in the exponential model
@everywhere function wmean(y, w)
    return sum(y .* w, 1)[:]
end

# The weighted exponential model by letif, as used in MCC experiments.
@everywhere function weighted_exponential_model(y_train, n_w::Integer)

    y_pred_mean = zeros(24, size(y_train)[2])
    for hour = 1:24
        y = y_train[hour:24:end]
        y_pred_mean[hour, :] = wmean(y, w(1:n_w*7, 3))
    end
    return y_pred_mean
end

# A simple quadratic model trained over the data, returning a 24 hour forecast
@everywhere function quadratic_model(x_train, x_test, y_train; train_hours = 12)

    # Prepare data
    x1 = float.(reshape(x_train[:time][end-(train_hours - 1):end], :, 1))
    x1_test = float.(reshape(x_test[:time], :, 1))
    y = disallowmissing(reshape(y_train[end-(train_hours - 1):end], :, 1))

    # Prepare the model
    f(θ) = θ[1] .+ θ[2] .* (x1) .+ θ[3] .* (x1).^2
    f_test(θ) = θ[1] .+ θ[2] .* (x1_test) .+ θ[3] .* (x1_test).^2
    obj(θ) = sum((f(θ) .- y).^2)  # Minimise the squared error
    res = GPForecasting.minimise(obj, [1.0, 1.0, 1.0], trace = false)

    return f_test(res)
end

@everywhere function quadratic_model_load(x_train, x_test, y_train; train_hours = 12)

    # Prepare data
    x1 = float.(reshape(x_train[:time][end-(train_hours - 1):end], :, 1))
    x2 = float.(reshape(x_train[:day_ahead_load_ID2][end-(train_hours - 1):end], :, 1))
    x1_test = float.(reshape(x_test[:time], :, 1))
    x2_test = float.(reshape(x_test[:day_ahead_load_ID2], :, 1))
    y = disallowmissing(reshape(y_train[end-(train_hours - 1):end], :, 1))

    # Prepare the model
    f(θ) = θ[1] .+ θ[2] .* (x1) .+ θ[3] .* (x1).^2 .+ θ[4] .* (x2) .+ θ[5] .* (x2).^2
    f_test(θ) = θ[1] .+ θ[2] .* (x1_test) .+ θ[3] .* (x1_test).^2 .+ θ[4] .* (x2_test) .+ θ[5] .* (x2_test).^2
    obj(θ) = sum((f(θ) .- y).^2)  # Minimise the squared error
    res = GPForecasting.minimise(obj, [1.0, 1.0, 1.0, 1.0, 1.0], trace = false)

    return f_test(res)
end

# A simple model that forecasts the previous day as the next
@everywhere function previous_day_model(y_train)
    y = disallowmissing(reshape(y_train, :, 1))
    return y[end-23:end]
end

# Temporary implementations
# regularised covariance matrix using the more sophisticated Ledoit-Wolf approach
@everywhere function cov_LW(X)
    n = size(X)[1]
    p = size(X)[2]

    S = cov(X, 1, false)
    Y = X .- mean(X, 1)
    m = trace(S) / p
    d2 = vecnorm(S .- m .* eye(S))^2 / p
    b2 = 0.0
    for k=1:n
        y = Y[k,:]
        b2 += vecnorm(y*y' .- S)^2
    end
    b2 = b2 / n^2 / p
    b2 = min(b2, d2)
    if b2 == d2
        println("WARNING: a2 is zero!")
    end
    a2 = d2 - b2
    return (b2 / d2 * m) .* eye(S) .+ (a2 / d2) .* S
end

@everywhere function mse(means, y_true)
    return mean((y_true .- means).^2)
end

@everywhere function mll_var(means, vars, y_true)
    return 0.5 * mean(log.(2π .* vars)) .+ 0.5 * mean((y_true .- means).^2 ./ vars)
end

@everywhere function mll_cov(means, covs, y_true)
    h = size(means)[1]
    n = size(means)[2]
    mll = 0.0
    for i=1:h
        C = covs[i,:,:]
        for j=1:n
            C[j,j] += GPForecasting._EPSILON_
        end
        L = chol(Symmetric(C))'
        z = L \ (y_true[i,:] .- means[i,:])
        mll += 0.5 * (n * log(2π) + 2.0 * sum(log.(diag(L))) + dot(z,z))
    end
    return mll / h
end

@everywhere function damec_exp(
    n_w::Int, # number of training weeks
    group::Int, # which group to run
    its::Int = 200, #Number of maximum optimization iterations
)

    info(GPForecasting.LOGGER, "LAM: n_w = ", n_w, " group = ", group)

    info(GPForecasting.LOGGER, "Grabbing data...")
    path = joinpath(GPForecasting.packagehomedir, "datasets")
    dat = mec_training_data(n_w*7, "DAMEC_train.csv", path, 1)

    # Create the Time Kernels
    k0 = 0.1 * stretch(EQ(), 12.01*2*2) * periodicise(stretch(EQ(), 0.51*10), Fixed(12.0))
    k1 = 0.4 * stretch(EQ(), 50.1*2.5*2) * periodicise(stretch(EQ(), 0.91*20), Fixed(24.0))
    k2 = 0.4 * stretch(EQ(), 50.1*2.5*2) * periodicise(stretch(EQ(), 1.01*7), Fixed(24.0*7))
    k3 = 0.05 * stretch(RQ(5.11), 1.01*5)
    k4 = 0.05 * stretch(MA(1/2), 1.01)
    kl1 = 2.1 * stretch(EQ(), 30.05)
    kl2 = 2.1 * stretch(EQ(), 30.05)
    kl3 = 2.1 * stretch(EQ(), 30.05)
    kl4 = 2.1 * stretch(EQ(), 30.05)

    # Define model
    κ₁ = (
        ((k0 + k1 + k2 + k3 + k4) ← :time) +
        (kl1 ← :day_ahead_load_ID2)
    )
    κ₂ = (
        ((k0 + k1 + k2 + k3 + k4) ← :time) *
        (kl1 ← :day_ahead_load_ID2)
    )
    κ₃ = (
        ((k0 + k1 + k2 + k3 + k4) ← :time) +
        (kl1 ← :day_ahead_load_ID1) +
        (kl2 ← :day_ahead_load_ID2) +
        (kl3 ← :day_ahead_load_ID3) +
        (kl4 ← :day_ahead_load_ID4)
    )
    κ₄ = (
        ((k0 + k1 + k2 + k3 + k4) ← :time) *
        (kl1 ← :day_ahead_load_ID1) *
        (kl2 ← :day_ahead_load_ID2) *
        (kl3 ← :day_ahead_load_ID3) *
        (kl4 ← :day_ahead_load_ID4)
    )
    κs = [κ₁, κ₂, κ₃, κ₄]

    # Run the experiment
    out = Dict(
                "posterior_metrics" => [],
                "comparison_model_metrics" => [],
                "hyper" => [],
                "runtime" => [],
                "opt_summary" => [],
                "kernel_index" => [],
                "split" => [],
            )

    info(GPForecasting.LOGGER, "Starting experiment...")
    for split in collect((group-1)*10+35:group*10+34).-(n_w-1)*7
        for q in 1:size(κs, 1) # Run each kernel on each group this time
            tic()
            info(GPForecasting.LOGGER, "Split $split, Kernel $(q)")

            # Clean up the x matrix from missing values
            for ξ in 1:size(dat[split]["train_x"], 2)
                dat[split]["train_x"][ξ] = disallowmissing(dat[split]["train_x"][ξ])
                dat[split]["test_x"][ξ] = disallowmissing(dat[split]["test_x"][ξ])
            end
            # Clean up the y matrix
            dat[split]["train_y"][1] = disallowmissing(dat[split]["train_y"][1])
            dat[split]["test_y"][1] = disallowmissing(dat[split]["test_y"][1])

            y_train = Matrix(dat[split]["train_y"]);
            y_test = Matrix(dat[split]["test_y"]);
            x_train = dat[split]["train_x"];
            x_test = dat[split]["test_x"];

            # Log the data
            log_y_train = log_transform(y_train);
            log_y_train_mean = mean(log_y_train, 1);
            log_y_train_std = std(log_y_train, 1);
            log_y_train_var = log_y_train_std.^2;

            info(GPForecasting.LOGGER, "Training GP...")
            gp = GP(κs[q]);
            θ_opt, gp = learn(gp, x_train, log_y_train .- log_y_train_mean, objective, its=its, trace=false, summary=true)

            pos = condition(gp, x_train, log_y_train .- log_y_train_mean);
            log_means = pos.m(x_test);
            log_vars = diag(pos.k(x_test));

            info(GPForecasting.LOGGER, "GP Trained, calculating metrics")
            # Inverse log transform, MC sample
            inverted_posterior_samples = inv_log_transform(sample(pos(x_test), 10000) .+ log_y_train_mean);
            inverted_μ = mean(inverted_posterior_samples, 2);
            inverted_cov = cov(inverted_posterior_samples');
            q95 = [quantile(inverted_posterior_samples[i, :], 0.95) for i in 1:size(inverted_posterior_samples, 1)]
            q05 = [quantile(inverted_posterior_samples[i, :], 0.05) for i in 1:size(inverted_posterior_samples, 1)]
            inverted_var = var(inverted_posterior_samples, 2);

            # Calculate alternative model metrics
            other_model_metrics = Dict{String, Any}(
                "mse_quad" => mse(quadratic_model(x_train, x_test, y_train, train_hours = 24*7), y_test),
                "mse_quad_load" => mse(quadratic_model_load(x_train, x_test, y_train, train_hours = 24*7*2), y_test),
                "mse_prev_day" => mse(previous_day_model(y_train), y_test),
                "mse_weighted" => mse(weighted_exponential_model(y_train, n_w), y_test),
            );

            # Calculate gp metrics
            posterior_metrics = Dict{String, Any}(
                "mse" => mse(inverted_μ, y_test),
                "mll_joint" => ModelAnalysis.mll_joint(inverted_μ[:], inverted_cov, y_test[:]),
                "mll_marginal" => ModelAnalysis.mll_marginal(inverted_μ[:], inverted_var[:], y_test[:]),
                "mll_joint_log" => ModelAnalysis.mll_joint(log_means, pos.k(x_test), (log_transform(y_test) .- log_y_train_mean)[:]),
                "mll_marginal_log" => ModelAnalysis.mll_marginal(log_means, log_vars[:], (log_transform(y_test) .- log_y_train_mean)[:]),
                "picp" => ModelAnalysis.picp(q05, q95, y_test[:]),
            );

            # Save results
            push!(out["posterior_metrics"], posterior_metrics);
            push!(out["comparison_model_metrics"], other_model_metrics);
            push!(out["hyper"], exp.(gp.k[:]));
            push!(out["runtime"], toc());
            push!(out["opt_summary"], θ_opt);
            push!(out["kernel_index"], q);
            info(GPForecasting.LOGGER, "Split $split, Kernel $q done!")
            println("Final results for split $split:")
            println("Posterior Metrics: $(out["posterior_metrics"][end])")
            println("Comparison Metrics: $(out["comparison_model_metrics"][end])")
        end
    end

    info(GPForecasting.LOGGER, "Experiment Complete")
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
function damec()
    parameters = [
        [3],
        [1],
        ]

    # -- Do not edit below this line -- #
    configuration = Dict{String, Any}(
        "parameters" => parameters,
        "experiment_function" => damec_exp,
        "seed" => 42,
    )
    return configuration
end
