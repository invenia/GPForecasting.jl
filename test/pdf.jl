@testset "pdf" begin
    gp = GP(0, 5 * ConstantKernel())
    x = collect(1:10)
    y = 18 * ones(10)
    obj = objective(gp, x, y)

    @test isa(obj, Function)
    @test isa(obj(GPForecasting.pack(Positive(18))), Float64)

    μ = [12., 3., 9.]
    Σ = Eye(3)
    n = Gaussian(μ, Σ)
    @test abs(logpdf(n, μ ./ 2) + 0.5*(log(det(Σ)) + 3log(2π) + (μ./2)' * Σ * μ./2)) < 1e-3
end
