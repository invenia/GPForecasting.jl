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

    @testset "Titsias" begin
        # 1D
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
        # 2D Input
        x = [x reverse(x)]
        y = sin.(4π * sum(x, dims=2)) .+ 1e-1 .* randn(size(x, 1))
        v1 = GPForecasting.logpdf(GP(periodicise(EQ(), 1.0) + 0.01 * DiagonalKernel()), x, y)
        Xm = collect(0:0.1:2);
        Xm = [Xm reverse(Xm)];
        sk = GPForecasting.SparseKernel(periodicise(EQ(), 1.0), Xm, Fixed(size(Xm, 1)), 0.01)
        v2 = GPForecasting.titsiasELBO(GP(sk), x, y)
        @test v2 <= v1 # v2 is the ELBO
        Xm = x
        sk = GPForecasting.SparseKernel(periodicise(EQ(), 1.0), Xm, Fixed(size(Xm, 1)), 0.01)
        v3 = GPForecasting.titsiasELBO(GP(sk), x, y)
        @test v3 ≈ v1 atol = 1e-8 # if Xm == x we should have ELBO == logpdf
        sgp = GP(sk)
        gp = GP(periodicise(EQ(), 1.0))
        v4 = GPForecasting.titsiasobj(gp, x, y, Xm, 0.01)(sgp.k[:])
        @test v4 ≈ -v1 atol = 1e-8   
        # OLMM
        y = [y 2y]
        H = [1.0 0.0; 0.0 1.0]
        k = periodicise(EQ(), 1.0)
        olmm_k = OLMMKernel(
            Fixed(2),
            Fixed(2),
            Fixed(1e-2),
            Fixed(1e-2),
            Fixed(H),
            Fixed(H),
            Fixed(H),
            Fixed([1.0, 1.0]),
            [k, k]
        )
        sgp = GP(GPForecasting.SparseKernel(olmm_k, Xm, 0.01))
        v1 = GPForecasting.titsiasELBO(sgp, x, y)
        Xm = collect(0:0.1:2);
        sgp = GP(GPForecasting.SparseKernel(olmm_k, Xm, 0.01))
        v2 = GPForecasting.titsiasELBO(sgp, x, y)
        k = periodicise(EQ(), 1.0) + 0.01 * DiagonalKernel()
        olmm_k.ks = [k, k]
        gp = GP(olmm_k)
        v3 = logpdf(gp, x, y)
        @test v3 ≈ v1 atol = 1e-8
        @test v1 >= v2
    end
end
