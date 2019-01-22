@testset "Experiments" begin

    @everywhere import Memento: getlogger
    configuration = GPForecasting.Experiments.experiment_template
    res = GPForecasting.experiment(configuration, trace = false)

    # Check that the experiment worked
    @test length(res) == 4 # Check that iterates through all parameters
    @test size(collect(values(res[1])), 1) == 3 # Check that all values are saved
    @test res[1]["parameters"] == (1680, 2)
    @test isa(res[1]["time"], AbstractFloat)
    @test isa(res[1]["output"], AbstractArray)
    for i in 2:size(res, 1)
        @test isa(res[i]["parameters"], Tuple{Int64,Int64})
        @test isa(res[i]["time"], AbstractFloat)
        @test isa(res[i]["output"], AbstractArray)
    end

    configuration = joinpath(
        GPForecasting.packagehomedir,
        "test/mock_experiment.jl"
    )
    res = GPForecasting.experiment(configuration, trace = false)

    # Check that the experiment worked
    @test length(res) == 4 # Check that iterates through all parameters
    @test size(collect(values(res[1])), 1) == 3 # Check that all values are saved
    @test res[1]["parameters"] == (168, 4)
    @test isa(res[1]["time"], AbstractFloat)
    @test isa(res[1]["output"], AbstractArray)
    for i in 2:size(res, 1)
        @test isa(res[i]["parameters"], Tuple{Int64,Int64})
        @test isa(res[i]["time"], AbstractFloat)
        @test isa(res[i]["output"], AbstractArray)
    end


    # Test the parsing
    new_args = String[
                "gpforecasting_job",
                "-c",
                "batch",
                "-n",
                "2",
                "-a",
                "s3",
                "-s",
                "test/experiment_test_script.jl",
                "--seed",
                "43",
                "-e",
                ":test"
            ]
    ref = Dict{String,Any}(
        "label" => "gpforecasting_job",
        "cluster-manager" => "batch",
        "seed" => 43,
        "nprocs" => 2,
        "archive" => AbstractString["s3"],
        "experiment-script" => "test/experiment_test_script.jl",
        "experiment" => ":test",
    )
    parsed = GPForecasting.arg_parse(new_args)
    @test all([(parsed[k] == ref[k]) for k in keys(parsed)])

    # Test the defaults
    empty_args = []
    ref = Dict{String,Any}(
          "label"             => "",
          "cluster-manager"   => "local",
          "seed"              => 42,
          "nprocs"            => 1,
          "archive"           => AbstractString[],
          "experiment-script" => nothing,
          "experiment" => nothing,
    )
    parsed = GPForecasting.arg_parse(empty_args)
    @test all([(parsed[k] == ref[k]) for k in keys(parsed)])

    # Test get parameters
    path = joinpath(GPForecasting.packagehomedir, "test/mock_experiment.jl")
    @test isa(GPForecasting.get_parameters(path), Dict{String, Any})
    @test GPForecasting.get_parameters(path)["seed"] == 42
    @test isa(GPForecasting.get_parameters(path)["experiment_function"], Function)
    @test isa(GPForecasting.get_parameters(path)["parameters"], Vector)

    # Test experiment when passing in a dictionary
    function dummy_fn(arg1, arg2)
        return arg1 + arg2
    end
    function dummy_set_parameters()

        # -- Edit Experiment Parameters to be varied -- #
        parameters = [
            [5],
            [2, 3],
        ]

        # -- Do not edit below this line -- #
        configuration = Dict{String, Any}(
            "parameters" => parameters,
            "experiment_function" => dummy_fn,
            "seed" => 422,
        )
        return configuration
    end
    configuration2 = dummy_set_parameters()
    res = GPForecasting.experiment(configuration2, trace = false)

    # Check that the experiment worked
    @test length(res) == 2 # Check that iterates through all parameters
    @test size(collect(values(res[1])), 1) == 3 # Check that all values are saved
    @test res[1]["parameters"] == (5, 2)
    @test res[2]["parameters"] == (5, 3)
    @test isa(res[1]["time"], AbstractFloat)
    @test isa(res[2]["time"], AbstractFloat)
    @test res[1]["output"] == 7
    @test res[2]["output"] == 8

    @test isa(
        GPForecasting.Experiments.describe(GPForecasting.Experiments.experiment_template),
        AbstractString
    )
    @test isa(
        GPForecasting.Experiments.source(GPForecasting.Experiments.experiment_template),
        AbstractString
    )
    @test isa(GPForecasting.Experiments.list_experiments(), Vector{Function})
end
