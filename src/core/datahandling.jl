using ModelAnalysis

export mcc_training_data, standardise_data, inverse_standardise, mec_training_data

"""
    mcc_training_data(
            training_days::Integer=30,
            filename::AbstractString="DF_train.csv",
            path::AbstractString=joinpath(GPForecasting.packagehomedir, "datasets"),
        )

# Arguments
- `training_days::Integer = 30`: The number of training days you would like to include in
    your training set.
- `filename`::AbstractString="DF_train.csv"
- `path::AbstractString=joinpath(GPForecasting.packagehomedir, "datasets")`:
Path to the data file.

# Returns
- `training_data::Array{Any, 1}`: An Array consisting of the train and test data. At each
index of the array you have a unique training set expressed as a dictionary:

```julia
Dict{String,DataFrames.DataFrame} with 4 entries:
  "train_y" => 720×2015 DataFrames.DataFrame.
  "train_x" => 720×8 DataFrames.DataFrame.
  "test_y"  => 24×2015 DataFrames.DataFrame.
  "test_x"  => 24×8 DataFrames.DataFrame.
```

Each "train_x" and "train_y" DataFrame will be of dimension `training_days * 24 x Z`, where
`Z` is the number of input features or output forecasts for multi-input and multi-output
models respectively. Each "test_x" and "test_y" will be of dimension `24 x Z`.

# Example
training_data = MCC_training_data()
x = training_data[1] # Take the first training set

p = GP(EQ() ← :time + EQ() ← :day_ahead_load_ID1)
p(x["train_x"]) # Evalue gp at the training x points
p(x["test_x"]) # Evaluate gp at testing x points

p(x["test_x"]).μ[:node_0688] - x["test_y"][:node_0688] # Subtracts the prediction from the
actual
"""
function mcc_training_data(
    training_days::Integer=30,
    filename::AbstractString="DF_train_v4.csv",
    path::AbstractString=joinpath(GPForecasting.packagehomedir, "datasets"),
    data::Symbol=:Delta,
    forecast_day::Integer=2,
)
    splits = ModelAnalysis.Data.MISO.load(
        training_days,
        filename,
        fdir=path,
        forecast_day=forecast_day
    )

    # Alter splits at every index, both for train_set and test_set
    training_data = []
    for i in 1:size(splits, 1)
        train_x, train_y = splits[i][1][1:end, 1:7], splits[i][1][1:end, 8:end]
        train_x[:time] = collect(1:1:size(train_x, 1)) # Add the time column
        test_x, test_y = splits[i][2][1:end, 1:7], splits[i][2][1:end, 8:end]
        test_x[:time] = collect(1:1:size(test_x, 1)) .+
            24 * (forecast_day - 1) .+ size(train_x, 1)# Add the time column
        if data == :DA || data == :RT
            s = data == :DA ? "_da" : "_rt"
            mask = endswith.(String.(names(train_y)), s)
            train_y = train_y[:, mask]
            new_names = [Symbol(replace(String(n), s => "")) for n in names(train_y)]
            names!(train_y, new_names)
            test_y = test_y[:, mask]
            names!(test_y, new_names)
        elseif data == :Delta
            maskda = endswith.(String.(names(train_y)), "_da")
            maskrt = endswith.(String.(names(train_y)), "_rt")
            train_da = train_y[:, maskda]
            train_rt = train_y[:, maskrt]
            test_da = test_y[:, maskda]
            test_rt = test_y[:, maskrt]
            nodes_da = [Symbol(replace(String(n), "_da" => "")) for n in names(train_da)]
            nodes_rt = [Symbol(replace(String(n), "_rt" => "")) for n in names(train_rt)]
            nodes_da != nodes_rt &&
                throw(error("Real time and day ahead node names don't match!"))
            deltas_train = Matrix(train_da) .- Matrix(train_rt)
            train_y = DataFrame(deltas_train, nodes_da)
            deltas_test = Matrix(test_da) .- Matrix(test_rt)
            test_y = DataFrame(deltas_test, nodes_da)
        else
            throw(ArgumentError("Unknown data option $data."))
        end
        push!(
            training_data,
            Dict(
                "train_x" => train_x,
                "train_y" => train_y,
                "test_x" => test_x,
                "test_y" => test_y,
            )
        )
    end
    return training_data
end

"""
    mec_training_data(
        training_days::Integer=30,
        filename::AbstractString="DAMEC_training.csv",
        path::AbstractString=joinpath(GPForecasting.packagehomedir, "datasets"),
        forecast_day::Integer=2,
    )

Load data for predicting MEC. Data set will be divided in several `split`s, corresponding to
the number of `training_days` specified. A test set will be built, corresponding to the
`forecast_day`th day after the training set.
"""
function mec_training_data(
    training_days::Integer=30,
    filename::AbstractString="DAMEC_training.csv",
    path::AbstractString=joinpath(GPForecasting.packagehomedir, "datasets"),
    forecast_day::Integer=2,
)
    splits = ModelAnalysis.Data.MISO.load(
        training_days,
        filename,
        fdir=path,
        forecast_day=forecast_day
    )

    training_data = []
    for i in 1:size(splits, 1)
        train_x, train_y = splits[i][1][1:end, 1:7], splits[i][1][1:end, 8:end]
        train_x[:time] = collect(1:1:size(train_x, 1)) # Add the time column
        test_x, test_y = splits[i][2][1:end, 1:7], splits[i][2][1:end, 8:end]
        test_x[:time] = collect(1:1:size(test_x, 1)) .+
            24 * (forecast_day - 1) .+ size(train_x, 1)# Add the time column
        push!(
            training_data,
            Dict(
                "train_x" => train_x,
                "train_y" => train_y,
                "test_x" => test_x,
                "test_y" => test_y,
            )
        )
    end
    return training_data
end

"""
    standardise_data(dat::Vector) -> Vector, Vector

Standardise data to zero mean and unitary variance. Returns the standardised data and also
the original means and standard deviations, to allow for the inverse transformation. Note
that the test data will be transformed according to the training data mean and std.
"""
function standardise_data(dat::Vector)
    cdat = deepcopy(dat)
    originals = sizehint!(Vector{Dict}(), length(dat))
    for d in cdat
        ytr = Matrix(d["train_y"])
        yte = Matrix(d["test_y"])
        stdtr = stddims(ytr, 1)
        meantr = meandims(ytr, 1)
        push!(originals, Dict(
                "mean_train" => meantr,
                "std_train" => stdtr,
            )
        )
        ytr = (ytr .- meantr) ./ stdtr
        yte = (yte .- meantr) ./ stdtr # Here we use the training data mean and std because
        # both sets should be transformed the same way.
        for i in 1:size(ytr, 2)
            d["train_y"][:, i] = ytr[:, i]
            d["test_y"][:, i] = yte[:, i]
        end
    end
    return cdat, originals
end

"""
    inverse_standardise(dat::Dict, orig::Dict) -> Dict
    inverse_standardise(dat::Vector, orig::Vector) -> Vector

Invert the standardisation of the data. This undos `standardise_data`.
"""
function inverse_standardise(dat::Dict, orig::Dict)
    test_y = orig["std_train"] .* Matrix(dat["test_y"]) .+ orig["mean_train"]
    train_y = orig["std_train"] .* Matrix(dat["train_y"]) .+ orig["mean_train"]
    out = deepcopy(dat)
    for i in 1:size(train_y, 2)
        out["train_y"][:, i] = train_y[:, i]
        out["test_y"][:, i] = test_y[:, i]
    end
    return out
end

function inverse_standardise(dat::Vector, orig::Vector)
    undat = sizehint!(Vector{Dict}(), length(dat))
    for (d, o) in zip(dat, orig)
        push!(undat, inverse_standardise(d, o))
    end
    return undat
end
