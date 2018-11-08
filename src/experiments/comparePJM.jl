function describe_comparePJM()
    d = """
        Compute LMM, OLMM and naive model for PJM.
        """
    println(d)
end

source_comparePJM() = "comparePJM.jl"

function comparePJM_exp(
    n_d::Int, # number of training days.
    n_dH::Int, # number of training days for H.
    m_olmm::Int, # number of latent processes.
    m_lmm::Int, # number of latent processes.
    σ²_olmm::Float64, # observation noise (note that we are working on the standardised space)
    σ²_lmm::Float64, # observation noise (note that we are working on the standardised space)
    lat_noise::Float64, # noise for the latent processes
    its::Int, # number of gradient descent steps.
    θ::Float64, # theta value
    opt_U::Bool, # Use greedy_U for the OLMM
    splits::Vector{Int}, # which splits of the data to run for.
    datapath::AbstractString = "", # path for the data.
    skip_stage::Vector{Bool} = [false, false, false] # PO, LMM, OLMM
)

    function wmean(y, w)
        return sum(y .* w, 1)[:]
    end

    dat = datapath != "" ?
        mcc_training_data(n_d, "PJM_train.csv", datapath) :
        mcc_training_data(n_d, "PJM_train.csv");

    Hdat = datapath != "" ?
        mcc_training_data(n_dH, "PJM_train.csv", datapath) :
        mcc_training_data(n_dH, "PJM_train.csv");

    # Truncate dat and translate it to coincide with Hdat
    d_cut = n_dH - n_d + 1
    dat = dat[d_cut:end];

    p = size(dat[1]["train_y"], 2) # Number of prices

    stdata, orig = standardise_data(dat); # Normalise data

    λ = 1.0 - exp(-1.0 / θ)
    weights = map(i -> λ * (1 - λ)^(1 - i), 1:n_d);
    weights /= sum(weights);

    # Define model
    # terms of time kernel
    k_time_1 = (0.16 * periodicise(MA(1/2) ▷  Fixed(0.1), Fixed(24.0)) * (RQ(2.0) ▷ Fixed(24.0 * 3.5)))
    k_time_2 = (0.16 * periodicise(MA(1/2) ▷  Fixed(0.5), Fixed(24.0)) * (RQ(2.0) ▷ Fixed(24.0 * 3.5)))
    k_time_3 = (0.16 * periodicise(MA(1/2) ▷  Fixed(0.1), Fixed(24.0)) * (RQ(2.0) ▷ Fixed(24.0 * 7.0)))
    k_time_4 = (0.16 * periodicise(MA(1/2) ▷  Fixed(0.5), Fixed(24.0)) * (RQ(2.0) ▷ Fixed(24.0 * 7.0)))
    k_time_5 = (0.16 * periodicise(MA(1/2) ▷  Fixed(0.1), Fixed(24.0)) * (RQ(2.0) ▷ Fixed(24.0 * 14.0)))
    k_time_6 = (0.16 * periodicise(MA(1/2) ▷  Fixed(0.5), Fixed(24.0)) * (RQ(2.0) ▷ Fixed(24.0 * 14.0)))

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
    k = (k_time ← :time) * (k_load ← :RTO)

    # Run the experiment
    out = Dict(
        :po_mse => [],
        :po_joint_mll => [],
        :po_marginal_mll => [],
        :olmm_mse => [],
        :olmm_joint_mll => [],
        :olmm_marginal_mll => [],
        :lmm_mse => [],
        :lmm_joint_mll => [],
        :lmm_marginal_mll => [],
        :po_time => [],
        :lmm_time => [],
        :olmm_time => [],
    )

    splits = splits == [-1] ? collect(1:length(dat)) : splits
    splits = splits[2] == -1 ? collect((length(dat) - splits[1] + 1):length(dat)) : splits
    info("Starting experiment...")
    for split in splits
        tic()
        info("Split $split...")
        sy_train = disallowmissing(Matrix(stdata[split]["train_y"]));
        y_train = disallowmissing(Matrix(dat[split]["train_y"]));
        y_trainH = disallowmissing(Matrix(Hdat[split]["train_y"]));
        y_test = disallowmissing(Matrix(dat[split]["test_y"])); # Let's test on untrasformed data.
        x_train = stdata[split]["train_x"];
        x_test = stdata[split]["test_x"];

        # Naive Model
        if !skip_stage[1]
            tic()
            means = zeros(24, p)
            covs = Matrix{Float64}[]
            for h = 1:24
                y = y_train[h:24:end, :]
                means[h,:] = wmean(y, weights)
                push!(covs, cov_LW(y))
            end

            gaus = Gaussian(
                Matrix{Float64}(means),
                BlockDiagonal(covs)
            )

            push!(out[:po_mse], ModelAnalysis.mse(gaus, y_test))
            push!(out[:po_joint_mll], ModelAnalysis.mll_joint(gaus, y_test))
            push!(out[:po_marginal_mll], ModelAnalysis.mll_marginal(gaus, y_test))
            push!(out[:po_time], toc())
        end

        # gaus = 0

        # # LMM
        if !skip_stage[2]
            tic()
            U, S, V = svd(cov_LW(y_trainH))
            m_lmm = min(m_lmm, p)
            H_lmm = U * diagm(sqrt.(S))[:, 1:m_lmm];
            σs = σ²_lmm .* var(y_train, 1)[:];
            gp = GP(LMMKernel(Fixed(m_lmm), Fixed(p), Positive(σs), Fixed(H_lmm), fill(k, m_lmm)));
            println("Time to start training the LMM")
            gp = learn(gp, x_train, (y_train .- orig[split]["mean_train"]), objective, its=its, trace=true);
            pos = condition(gp, x_train, (y_train .- orig[split]["mean_train"]));
            gaus = pos(x_test; hourly = true)
            gaus.μ = gaus.μ .+ orig[split]["mean_train"];

            push!(out[:lmm_mse], ModelAnalysis.mse(gaus, y_test))
            push!(out[:lmm_joint_mll], ModelAnalysis.mll_joint(gaus, y_test))
            push!(out[:lmm_marginal_mll], ModelAnalysis.mll_marginal(gaus, y_test))
            push!(out[:lmm_time], toc())
        end
        # # gaus = 0
        # println("LMM done")
        # OLMM
        if !skip_stage[3]
            tic()
            stds = orig[split]["std_train"]' * orig[split]["std_train"];
            U, S, V = svd(cov_LW(
                (y_trainH .- orig[split]["mean_train"]) ./ orig[split]["std_train"]
            ))
            m_olmm = min(m_olmm, p)
            H_olmm = U * diagm(sqrt.(S))[:, 1:m_olmm];
            S_sqrt = sqrt.(diag(H_olmm' * H_olmm));
            U = H_olmm * diagm(S_sqrt.^(-1.0));
            _, P = GPForecasting.build_H_and_P(U, S_sqrt);
            gp = GP(OLMMKernel(
                Fixed(m_olmm),
                Fixed(p),
                Positive(σ²_olmm),
                Positive(lat_noise),
                Fixed(H_olmm),
                Fixed(P),
                Fixed(U),
                Fixed(S_sqrt),
                [k for i in 1:m_olmm]
            ));
            println("Time to start training the OLMM")
            gp = learn(gp, x_train, sy_train, objective, its=its, trace=true, opt_U=opt_U);
            pos = condition(gp, x_train, sy_train);
            gaus = pos(x_test; hourly = true)

            # Un-normalise the predictions
            gaus.μ = orig[split]["std_train"] .* gaus.μ .+ orig[split]["mean_train"];
            gaus.Σ = BlockDiagonal([Hermitian(b .* stds) for b in blocks(gaus.Σ)]);

            push!(out[:olmm_mse], ModelAnalysis.mse(gaus, y_test))
            push!(out[:olmm_joint_mll], ModelAnalysis.mll_joint(gaus, y_test))
            push!(out[:olmm_marginal_mll], ModelAnalysis.mll_marginal(gaus, y_test))
            push!(out[:olmm_time], toc())
        end
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
function comparePJM()
    parameters = [
        [3 * 7],
        [30],
        [48],
        [20],
        [0.15],
        [0.15],
        [5.0],
        [15],
        [21. / 4.],
        [false],
        [[1]],
        [""],
        [[false, false, false]]
        ]

    # -- Do not edit below this line -- #
    configuration = Dict{String, Any}(
        "parameters" => parameters,
        "experiment_function" => comparePJM_exp,
        "seed" => 42,
    )
    return configuration
end
