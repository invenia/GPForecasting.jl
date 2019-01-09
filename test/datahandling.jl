@testset "DataHandling" begin

    training_data = mcc_training_data(
        1,
        "sample.csv",
        joinpath(GPForecasting.packagehomedir, "test"),
        :DA
    )
    # rm(joinpath(GPForecasting.packagehomedir, "test", "sample.csv")) # Removing this here
    # because CI is not able to download the file from S3. We should test manually that it
    # works for now.
    @test size(training_data[1]["train_x"], 1) == 24
    @test size(training_data[2]["train_x"], 2) == 8
    @test size(training_data[1]["train_y"], 1) == 24
    @test size(training_data[1]["test_x"], 1) == 24
    @test size(training_data[2]["test_x"], 2) == 8
    @test size(training_data[1]["test_y"], 1) == 24
    cdat, orig = standardise_data(training_data)
    @test meandims(Matrix(cdat[1]["train_y"]), 1) ≈ zeros(size(cdat[1]["train_y"], 2))' atol = _ATOL_
    @test stddims(Matrix(cdat[1]["train_y"]), 1) ≈ ones(size(cdat[1]["train_y"], 2))' atol = _ATOL_
    idat = inverse_standardise(cdat, orig)
    @test Matrix(idat[1]["test_y"]) ≈ Matrix(training_data[1]["test_y"]) atol = _ATOL_
    @test Matrix(idat[1]["train_y"]) ≈ Matrix(training_data[1]["train_y"]) atol = _ATOL_
    @test Matrix(idat[end]["test_y"]) ≈ Matrix(training_data[end]["test_y"]) atol = _ATOL_
    @test Matrix(idat[end]["train_y"]) ≈ Matrix(training_data[end]["train_y"]) atol = _ATOL_
    rt = mcc_training_data(
        1,
        "sample.csv",
        joinpath(GPForecasting.packagehomedir, "test"),
        :RT
    )
    deltas = mcc_training_data(
        1,
        "sample.csv",
        joinpath(GPForecasting.packagehomedir, "test"),
        :Delta
    )
    @test Matrix(deltas[1]["train_y"]) ≈ Matrix(training_data[1]["train_y"]) .- Matrix(rt[1]["train_y"])

    mec_data = mec_training_data(1, "DAMEC_sample.csv", joinpath(GPForecasting.packagehomedir, "test"), 1)
    @test length(mec_data) == 1
    @test all(in(keys(mec_data[1])), ["test_x", "test_y", "train_x", "train_y"])
end
