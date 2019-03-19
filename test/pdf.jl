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

    x = collect(0:0.01:2)
    y = sin.(4π * x) .+ 1e-1 .* randn(length(x))
    v1 = GPForecasting.logpdf(GP(periodicise(EQ(), 1.0) + 0.01 * DiagonalKernel()), x, y)
    Xm = collect(0:0.1:2);
    sk = GPForecasting.SparseKernel(periodicise(EQ(), 1.0), Xm, Fixed(length(Xm)), 0.01)
    v2 = GPForecasting.titsiasELBO(GP(sk), x, y)
    @test v2 <= v1 # v2 is the ELBO
    Xm = x
    sk = GPForecasting.SparseKernel(periodicise(EQ(), 1.0), Xm, Fixed(length(Xm)), 0.01)
    v3 = GPForecasting.titsiasELBO(GP(sk), x, y)
    @test v3 ≈ v1 atol = 1e-8 # if Xm == x we should have ELBO == logpdf
    sgp = GP(sk)
    gp = GP(periodicise(EQ(), 1.0))
    v4 = GPForecasting.titsiasobj(gp, x, y, Xm, 0.01)(sgp.k[:])
    @test v4 ≈ -v1 atol = 1e-8
end
