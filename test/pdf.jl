@testset "pdf" begin
    gp = GP(0, 5 * ConstantKernel())
    x = collect(1:10)
    y = 18 * ones(10)
    obj = mle_obj(gp, x, y)

    @test isa(obj, Function)
    @test isa(obj(GPForecasting.pack(Positive(18))), Float64)

    μ = [12., 3., 9.]
    Σ = Eye(3)
    n = Gaussian(μ, Σ)
    @test abs(logpdf(n, μ ./ 2) + 0.5*(log(det(Σ)) + 3log(2π) + (μ./2)' * Σ * μ./2)) < 1e-3

    reg(gp, x, y) = sum(gp.k(x))
    @test logpdf(gp, x, y, gp.k[:]) - reg(gp, x, y) ≈ reglogpdf(reg, gp, x, y, gp.k[:]) atol = _ATOL_ rtol = _RTOL_

    # OLMM
    A = rand(5, 3);
    U, S, V = svd(A);
    H = U * Diagonal(S);
    gp = GP(OLMMKernel(3, 5, 1e-2, 1e-2, H, [EQ() ▷ Fixed(1.0), EQ() ▷ Fixed(2.0), EQ() ▷ Fixed(3.0)]));
    y = sample(gp(x));
    reg(gp, x, y) = sum(abs.(gp.k.H))
    @test logpdf(gp, x, y, gp.k[:]) - reg(gp, x, y) ≈ reglogpdf(reg, gp, x, y, gp.k[:]) atol = _ATOL_ rtol = _RTOL_
    @test mle_obj(gp, x, y)(gp.k[:]) + reg(gp, x, y) ≈ map_obj(reg, gp, x, y)(gp.k[:]) atol = _ATOL_ rtol = _RTOL_
    @test map_obj(reg, gp, x, y)(gp.k[:]) ≈ map_obj(reg)(gp, x, y)(gp.k[:]) atol = _ATOL_ rtol = _RTOL_

    x_train, x_val = x[1:5], x[6:10]
    y_train, y_val = y[1:5, :], y[6:10, :]
    pos = condition(gp, x_train, y_train)
    mll = sum([-logpdf(Gaussian(pos.m(x_val[i, :])[:], pos.k(x_val[i, :])), y_val[i, :]) for i=1:5])
    @test mll_block_posterior_obj(gp, x_train, y_train, x_val, y_val)(gp.k[:]) ≈ mll atol = _ATOL_ rtol = _RTOL_

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
        y = sin.(4π * sum(x, dims=2)) .+ 1e-1 .* randn(size(x, 1)); y = dropdims(y, dims=2)
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
        x = collect(0:0.01:2)
        y = sin.(4π * x) .+ 1e-1 .* randn(length(x))
        y = [y 2y]
        Xm = x;
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

@testset "Expected return" begin
    gp = GP(EQ())
    gp2 = GP(1, EQ())
    x = [1]
    y = [10]
    @test normalised_expected_return(gp, x, y) <= normalised_expected_return(gp2, x, y)
    @test normalised_expected_return(gp, x, y) == normalised_expected_return_balanced(gp, x, y)
    @test normalised_expected_return_balanced(gp2, x, y) <= normalised_expected_return(gp2, x, y)
    x = rand(3)
    y = 10 .* ones(3, 1)
    @test normalised_expected_return(gp, x, y) <= normalised_expected_return(gp2, x, y)
    m = 2; p = 3; σ² = 0.1; lat_noise = 0.1
    U, S, V = svd(rand(p, p))
    H = U * Diagonal(sqrt.(S))[:, 1:m]
    S_sqrt = sqrt.(diag(H' * H))
    U = H * Diagonal(S_sqrt.^(-1.0))
    _, P = GPForecasting.build_H_and_P(U, S_sqrt)
    k = 1.0 * (EQ() ▷  1.0) + 2.0
    gp = GP(OLMMKernel(
           Fixed(m),
           Fixed(p),
           Positive(σ²),
           Positive(lat_noise),
           H,
           Fixed(P),
           Fixed(U),
           Positive(S_sqrt),
           [k for i in 1:m]
      ))
      yc = rand(5, 3)
      xc = rand(5)
      yt = rand(4, 3)
      xt = rand(4)
      f = normalised_expected_posterior_return_obj(gp, xc, xt, yc, yt)
      g = normalised_expected_posterior_return_balanced_obj(gp, xc, xt, yc, yt)
      @test f(gp[:]) <= g(gp[:])
      grad = ∇(f)(gp[:])[1]
      grad2 = ∇(g)(gp[:])[1]
      @test grad != grad2
      # Manual gradient step
      @test f(gp[:] - 1e-6 * grad) < f(gp[:])
      k = (1.0 * (EQ() ▷  1.0) + 2.0) ← :input
      gp = GP(OLMMKernel(
             Fixed(m),
             Fixed(p),
             Positive(σ²),
             Positive(lat_noise),
             H,
             Fixed(P),
             Fixed(U),
             Positive(S_sqrt),
             [k for i in 1:m]
        ))
        dfc = DataFrame([[1.,2.,3.], [1.,1.,1.]], [:input, :input2])
        dft = DataFrame([[1.1,2.1,3.1], [1.,1.,1.]], [:input, :input2])
        f = normalised_expected_posterior_return_obj(gp, dfc, dft, rand(3, 3), rand(3, 3))
        grad = ∇(f)(gp[:])[1]
        # Manual gradient step
        @test f(gp[:] - 1e-6 * grad) < f(gp[:])
        @test g(gp[:] - 1e-6 * grad2) < g(gp[:])
end
