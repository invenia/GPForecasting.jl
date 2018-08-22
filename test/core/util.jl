using CSV

@testset "Util" begin
    data = CSV.read(joinpath(GPForecasting.packagehomedir, "test", "core", "cov.csv"), header=false)
    X = convert.(Float64, Matrix(data[1:20,:]))
    C = convert.(Float64, Matrix(data[21:end,:]))

    @test cond(cov(X)) > cond(cov_EB(X)) > cond(cov_LW(X))
    @test cov_LW(X) ≈ C atol = _ATOL_
end
