export GPForecaster

# This contains the forecaster for Simulation.jl
using Base.Dates
using AutoHashEquals
using DataFrames
using DataStructures
using Intervals
using Missings
using TimeZones
using StatsBase

# Private deps
using BidDatabase
using DataFeatures
using DateOffsets
using ElectricityMarkets
using S3DB
using Forecasters
import Simulation: SIM_NOW_OFFSET, collection

# imports
import IterTools: product
import KeyedFrames: KeyedFrame

const MISSING_DATA_THR = 0.1 # Threshold to drop a node due to excessive misisng data.

@auto_hash_equals struct GPForecaster{G<:Grid, T<:Kernel} <: AbstractForecaster
    market::Market{G}
    gp::GP{T, <:Mean}
    training_window::Range # Range to get training data for
    H_window::Range # Range to get data for building the mixing matrix
    standardise::Bool # Standardise prices (zero mean, unitary variance)
    learning_its::Int
    initialiseH::Bool
end

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
k = (k_time ← :time) * (k_load ← :load)

function GPForecaster(
    market::Market{G},
    kernel_type::Type{T};
    m=30, # Number of latent processes.
    training_window=Day(2):Day(1):Day(3 * 7 + 1),
    H_window=Day(2):Day(1):Day(3 * 30 + 1),
    standardise=false,
    learning_its=25,
    initialiseH=true,
) where {G<:Grid, T<:LMMKernel}
    # ===== defaults ======
    p = 1 # Number of nodes to predict. Again, an issue with the workflow order. Should be
    # updated after data is fetched.
    σ² = [-1.0] # Same as above. Here it is more critical because we might want to enforce a
    # value, instead of just getting it from data.
    H = Matrix{Float64}(0, 0)
    return GPForecaster(
        market,
        GP(LMMKernel(m, p, σ², H, k)),
        training_window,
        H_window,
        standardise,
        learning_its,
        initialiseH,
    )
end

function GPForecaster(
    market::Market{G},
    kernel_type::Type{T};
    m=300, # Number of latent processes.
    σ²=0.1,
    lat_noise=5.0,
    training_window=Day(2):Day(1):Day(3 * 7 + 1),
    H_window=Day(2):Day(1):Day(3 * 30 + 1),
    standardise=true,
    learning_its=0,
    initialiseH=true,
) where {G<:Grid, T<:OLMMKernel}
    # ===== defaults ======
    p = -1 # Number of nodes to predict. Again, an issue with the workflow order. Should be
    # updated after data is fetched.
    H = Matrix{Float64}(0, 0)
    return GPForecaster(
        market,
        GP(OLMMKernel(
            Fixed(m),
            Fixed(p),
            Fixed(σ²),
            Fixed(lat_noise),
            Fixed(H),
            Fixed(Matrix{Float64}(0, 0)),
            Fixed(Matrix{Float64}(0, 0)),
            Fixed(Vector{Float64}(0)),
            [k for i in 1:m],
        )),
        training_window,
        H_window,
        standardise,
        learning_its,
        initialiseH,
    )
end

function Base.show(io::IO, f::GPForecaster{G, T}) where {G<:Grid, T<:Kernel}
    print(io, "GPForecaster(Market{$G}, Process{$T})")
end

function DateOffsets.targets(f::GPForecaster, sim_now)
    DateOffsets.targets(Horizon(; span=Day(1), step=Hour(1)), sim_now)
end
TimeZones.timezone(f::GPForecaster) = timezone(f.market)
collection(f::GPForecaster) = lowercase(grid_name(f.market))

function offsets(f::GPForecaster)
    tw = f.training_window
    hw = f.H_window
    # Offsets: windows are given in days. This means that one has to add one hour at the beginning, and 24 at the end.
    # This is likely a place that requires double/triple checks as cheating might be involved
    twoffs = SIM_NOW_OFFSET .+ StaticOffset.((Hour(-tw[1]) - Hour(1)):Hour(-1):Hour(-tw[end] - Day(1)))
    hwoffs = SIM_NOW_OFFSET .+ StaticOffset.((Hour(-tw[1]) - Hour(1)):Hour(-1):Hour(-hw[end] - Day(1)))
    return Dict(
        :training_window_offsets => twoffs,
        :H_window_offsets => hwoffs,
    )
end

function query(
    forecaster::GPForecaster,
    client::S3DB.AbstractClient,
    sim_now::ZonedDateTime,
    nodes::AbstractArray{<:AbstractString}
)
    # Mostly stealing the EmpiricalForecaster code. Got to double check everything
    queries = Dict{Tuple{Symbol, Symbol}, Any}()
    data_features = Dict{Symbol, Any}()

    # horizon = Horizon(; span=Day(1), step=Hour(1))
    horizon = Horizon(; span=Day(1), step=Day(1))
    horizon_h = Horizon(; span=Day(1), step=Hour(1))

    # Currently cannot do delta.
    market_timings = [:realtime, :dayahead]
    tags = ["HTTPFinal", "HTTP"] # These seem to be specific to MISO

    # These are needed for data grabbing
    the_offsets = offsets(forecaster)

    # Define features.
    for (market_timing, tag) in zip(market_timings, tags)
        lmp = DataSource(client, collection(forecaster), "$(market_timing)_price")

        data_features[:Hdata] = DataFeature(
            lmp,
            Dict(:node => nodes),
            the_offsets[:H_window_offsets];
            availability=:cheating,
        )
        data_features[:lmp_train] = DataFeature(
            lmp,
            Dict(:node => nodes),
            the_offsets[:training_window_offsets];
            availability=:cheating,
        )

        queries[(:Hdata, market_timing)] = FeatureQuery(
            data_features[:Hdata], horizon, sim_now,
        )
        feature_names = [:lmp_train]
        for k in feature_names
            queries[(k, market_timing)] = FeatureQuery(
                data_features[k], horizon, sim_now,
            )
        end
    end

    # We need to grab only dayahead load
    load_ds = DataSource(client, collection(forecaster), "dayahead_load")
    data_features[:load_train] = DataFeature(
        load_ds,
        Dict(:region => ["MISO"]);
        availability=:cheating,
    )
    data_features[:load_predict] = DataFeature(
        load_ds,
        Dict(:region => ["MISO"]),
    )

    queries[(:load_train, :dayahead)] = FeatureQuery(
        data_features[:load_train], horizon_h, sim_now, forecaster.training_window
    )
    queries[(:load_predict, :dayahead)] = FeatureQuery(
        data_features[:load_predict], horizon_h, sim_now
    )

    return queries
end

function Base.fetch(
    forecaster::GPForecaster,
    client::S3DB.AbstractClient,
    sim_now::ZonedDateTime,
    nodes::AbstractArray{<:AbstractString}
)

    # Query for all of our data
    queries = query(forecaster, client, sim_now, nodes)

    # Fetch our results from S3DB
    features = map(queries) do query_pair
        data_name, data_query = query_pair
        return data_name => fetch(data_query)
    end

    # features[:nodes] = nodes # This does not work since
    # features is Dict{Tuple{Symbol,Symbol},KeyedFrame}
    features[(:nodes, :all)] = KeyedFrame(DataFrame(:nodes => nodes), :nodes)

    return features
end

function get_delta(df_da::AbstractDataFrame, df_rt::AbstractDataFrame, col::Symbol)
    tmp = deepcopy(df_rt)
    tmp[col] = df_da[col] .- df_rt[col]
    return tmp
end

function standardise_data(dat::AbstractDataFrame)
    ytr = Matrix{Float64}(dat)
    stdtr = std(ytr, 1)
    meantr = mean(ytr, 1)
    originals = Dict("mean" => meantr, "std" => stdtr)
    ytr = (ytr .- meantr) ./ stdtr
    return ytr, originals
end

function inverse_standardise(mvn::MvNormal, orig::Dict)
    stds = orig["std"]' * orig["std"]
    means = orig["std"][:] .* mean(mvn)[:] .+ orig["mean"][:] # reshaping
    # because MvNormal breaks otherwise.
    covs = cov(mvn) .* stds # HERE WE ASSUME COVARIANCE FOR A SINGLE HOUR! Will need to
    # dispatch for joint covariances (or use a size check)!
    return MvNormal(means, covs)
end

function transform(
    forecaster::GPForecaster,
    features::Dict{Tuple{Symbol,Symbol},<:AbstractDataFrame},
)

    market_timings = [:dayahead, :realtime]
    # Make sure everything is sorted
    for f in product([:lmp_train, :Hdata], market_timings)
        sort!(features[f], [:validity, :node])
    end
    for f in product([:load_train, :load_predict], [:dayahead]) # Load is available only in DA
        sort!(features[f], [:validity]) # no node info here
    end

    nodes = features[(:nodes, :all)][:nodes]

    # Get the deltas
    cols = Dict(
        :Hdata => :lmp,
        :lmp_train => :lmp,
    )
    deltas = map(cols) do kvpair
        k, v = kvpair
        return k => get_delta(features[(k, :dayahead)], features[(k, :realtime)], v)
    end

    # Determine the beginning and end for both train and predict
    train_tgs = features[(:load_train, :dayahead)][:target]
    predict_tgs = features[(:load_predict, :dayahead)][:target]
    train_start = 1 # We will just number the hours, so this is our one.
    train_end = convert(Int64, Dates.Hour(train_tgs[end] - train_tgs[1])) + 1
    predict_start = convert(Int64, Dates.Hour(predict_tgs[1] - train_tgs[1])) + 1
    predict_end = convert(Int64, Dates.Hour(predict_tgs[end] - train_tgs[1])) + 1
    debug(
        LOGGER,
        "There is a gap of $(predict_start - train_end) hours between training and predicting"
    )

    # Organise everything as a timestamp vs node/load/etc. dataframe
    train = Dict{Symbol, Any}() # This is set to Any to avoid unintended automatic conversion accidents
    predict = Dict{Symbol, Any}() # This is set to Any to avoid unintended automatic conversion accidents
    nodal_qts = [:Hdata, :lmp_train]
    for qt in nodal_qts
        df = DataFrame()
        # Will build stuff column by column. Might not be the most efficient way to go about it.
        for n in nodes
            mask = deltas[qt][:node] .== n
            df[Symbol(n)] = deltas[qt][mask, :lmp]
        end
        train[qt] = df
    end
    # global_qts = [:load]
    train[:input] = DataFrame()
    predict[:input] = DataFrame()
    train[:input][:load] = features[(:load_train, :dayahead)][:load]
    predict[:input][:load] = features[(:load_predict, :dayahead)][:load]

    # TODO: reinstate for loop to guarantee automatic generalizability

    # Add :time column
    train[:input][:time] = collect(train_start:train_end)
    predict[:input][:time] = collect(predict_start:predict_end)

    # Add the :target column
    predict[:input][:target] = predict_tgs
    # out[:train][:target] = train_tgs

    # Get rid of missings
    to_delete = []
    for k in nodal_qts
        # Naive check to see if we got a new node that has no backfilled data. In this case,
        # we'll pretend it does not exist. THIS IS PROBABLY NOT WHAT WE'D REALLY WANT.
        # Backfilling should be the correct way to go.
        for n in names(train[k])
            sum(ismissing.(train[k][n])) / length(train[k][n]) > MISSING_DATA_THR &&
                push!(to_delete, n)
        end
        # Adding a little more background to this: we use the price data in two different
        # ways: 1- To build the nodal covariance matrix, which we use to obtain the mixing
        # matrix H. 2- To condition the observations during the `fit` phase. We may be able
        # to do `1` with some missing data and still obtain decent results, but it is not
        # possible to condition on the observations over only a subset of the nodes, i.e.,
        # every timestamp that is used for the training/fit phase has to have data on all
        # nodes. Moreover, the number of nodes has to be the same in the mixing matrix, in
        # the training/fit phase and in the prediction phase.
    end
    to_delete = unique(to_delete)

    # Track deleted nodes
    train[:deleted_nodes] = DataFrame(to_delete)

    for k in nodal_qts
        for n in to_delete
            warn("Removing node $n due to missing data.")
            delete!(train[k], n)
        end
        # This only works well if we have few eventual missing values. In case we have a new
        # node showing up without having backfilled data, we'll basically destroy all
        # data.
        dropmissing!(train[k])
    end

    predict[:nodes] = names(train[:lmp_train])
    train[:nodes] = predict[:nodes]

    # Build covariance for H
    train[:covd] = GPForecasting.cov_LW(Matrix{Float64}(train[:Hdata])) # NOTE: depending on the type of out,
    # there could be issues like automatic promotion to DataFrame. At this stage, this is supposed
    # to be a Matrix, but keep in mind this.
    delete!(deltas, :Hdata) # We don't need this anymore

    if forecaster.standardise
        std_data, origs = standardise_data(train[:lmp_train])
        for i in size(train[:lmp_train], 2)
            train[:lmp_train][:, i] = std_data[:, i]
        end
        predict[:origs] = origs
        # got to also rescale the covariance for computing H
        stds = origs["std"]' * origs["std"]
        train[:covd] = train[:covd] ./ stds
    end

    return train, predict
end

function StatsBase.fit(
    forecaster::GPForecaster,
    features::Dict{Symbol, <:Any},
)
    m = GPForecasting.unwrap(forecaster.gp.k.m)
    gp = deepcopy(forecaster.gp)
    gp.k.p = GPForecasting.Fixed(length(features[:nodes]))
    x_train = features[:input]
    y_train = Matrix{Float64}(features[:lmp_train])
    if isa(gp.k, LMMKernel)
        gp.k.σ² = GPForecasting.Positive(0.1 * var(y_train, 1)[:])
    end
    if forecaster.initialiseH
        debug(LOGGER, "Initialising mixing matrix.")
        U, S, V = svd(features[:covd])
        H = U * diagm(sqrt.(S))[:, 1:m]
        gp.k.H = typeof(gp.k.H)(H) # `typeof` ensures we keep constraints ; This is H, drop the type stuff
        if isa(forecaster.gp.k, OLMMKernel)
            # This is necessary here because it is usually done by the GP constructor. There
            # is an issue with the order of the workflow.
            S_sqrt = sqrt.(diag(H' * H))
            U = H * diagm(S_sqrt.^(-1.0))
            _, P = GPForecasting.build_H_and_P(U, S_sqrt)
            gp.k.S_sqrt = typeof(gp.k.S_sqrt)(S_sqrt) # Type of this matrix needs to be checked
            gp.k.U = typeof(gp.k.U)(U) # Type of this matrix needs to be checked
            gp.k.P = typeof(gp.k.P)(P) # Type of this matrix needs to be checked
        end
    end
    if forecaster.learning_its > 0
        gp = GPForecasting.learn(gp, x_train, y_train, objective, its=forecaster.learning_its) # there were kwargs here but they can't be passed from simulate
    end
    pos = GPForecasting.condition(gp, x_train, y_train)
    return GPForecaster(
        forecaster.market,
        pos,
        forecaster.training_window,
        forecaster.H_window,
        forecaster.standardise,
        forecaster.learning_its,
        forecaster.initialiseH,
    )
end

function StatsBase.predict(
    forecaster::GPForecaster,
    features::Dict{Symbol, <:Any},
)
    pred_features = features[:input]
    node_list = features[:nodes]
    mvn = GPForecasting.MvNormal(forecaster.gp, pred_features)
    # transform back to the original space in case there standardisation
    if forecaster.standardise
        mvn = inverse_standardise(mvn, features[:origs])
    end
    return IndexedDistribution(mvn, node_list)
end

function StatsBase.predict(
    forecaster::GPForecaster,
    features::Dict{Symbol, <:Any},
    targets::Vector,
)
    mask = Bool.(sum([features[:input][:target] .== t for t in targets])) # ugly hack
    pred_features = features[:input][mask, :]
    new_features = deepcopy(features)
    new_features[:input] = pred_features
    return predict(forecaster, new_features)
end

function StatsBase.predict(
    forecaster::GPForecaster,
    features::Dict{Symbol, <:Any},
    targets,
)
    mask = features[:input][:target] .== targets
    pred_features = features[:input][mask, :]
    new_features = deepcopy(features)
    new_features[:input] = pred_features
    return predict(forecaster, new_features)
end
