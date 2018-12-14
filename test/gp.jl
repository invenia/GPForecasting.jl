@testset "GP" begin
    gp = GP(0, EQ())
    x = collect(1.0:10.0)
    y = 2 .* x

    ngp = 2 * gp
    @test ngp.m(x) ≈ 2 .* gp.m(x) atol = _ATOL_
    @test ngp.k(x) ≈ 4 .* gp.k(x) atol = _ATOL_
    @test (ngp + gp).m(x) ≈ ngp.m(x) + gp.m(x) atol = _ATOL_
    @test (ngp + gp).k(x) ≈ ngp.k(x) + gp.k(x) atol = _ATOL_

    mvn = MvNormal(gp, x)
    @test isa(mvn, MvNormal)
    @test size(mvn.μ) == (10,)
    @test size(mvn.Σ) == (10, 10)

    posterior = condition(gp, x, y)

    @test isa(posterior, GP)
    @test isa(posterior.m, Mean)
    @test isa(posterior.k, Kernel)

    @test abs(posterior.m([5.])[1] - 10.0) < 0.1
    @test sum(posterior.k([1.,2.,4.],[1.,2.,4.])) ≈ 2.9999903579658316e-6 atol = _ATOL_
    @test 9 < sample(posterior([5.,6.]))[1] < 11
    @test isa(sample(posterior([5.,6.])), AbstractArray)
    @test all(
        credible_interval(posterior, [1.,5.,50.])[2] .<
        credible_interval(posterior, [1.,5.,50.])[1] .<
        credible_interval(posterior, [1.,5.,50.])[3]
    )

    @testset "Get/Set for GPs" begin
        p1 = GP(ConstantMean(1.0) + ConstantMean(Fixed(5.0)), stretch(EQ(), 2.0))
        p2 = GP(FunctionMean(sin) * ConstantMean(Positive(10.0)), stretch(EQ(), 6.0))
        p3 = p1 + p2

        @test GPForecasting.get(p1) ≈ [1.0, log(2.0)] atol = _ATOL_
        @test GPForecasting.get(p2) ≈ [log(10.0), log(6.0)] atol = _ATOL_
        @test GPForecasting.get(p3) ≈ [
                                        GPForecasting.get(p1.m);
                                        GPForecasting.get(p2.m);
                                        GPForecasting.get(p1.k);
                                        GPForecasting.get(p2.k);
                                      ] atol = _ATOL_

        @test GPForecasting.get(GPForecasting.set(p1, [5.0, 5.0])) ≈ [5.0, 5.0] atol = _ATOL_
        @test GPForecasting.get(GPForecasting.set(p2, [5.0, 5.0])) ≈ [5.0, 5.0] atol = _ATOL_
        @test GPForecasting.get(GPForecasting.set(p3, [5.0, 5.0, 6.0, 7.0])) ≈ [5.0, 5.0, 6.0, 7.0] atol = _ATOL_
        # Explicitly test setting
        p4 = GPForecasting.set(p3, [5.0, 5.0, 6.0, 7.0])
        @test p4[:] ≈ [5.0, 5.0, 6.0, 7.0] atol = _ATOL_

        # The following tested against the results of GPFlow
        @test p3.m([1.0, 2.0, 3.0]) ≈ [14.414709848078965, 15.092974268256818, 7.411200080598672] atol = _ATOL_
        @test p3.k([1.0, 2.0, 3.0]) ≈ [2.         1.86870402 1.55249013;
                                       1.86870402 2.         1.86870402;
                                       1.55249013 1.86870402 2.        ] atol = _ATOL_

    end

    @testset "Multi-dimensional GPs" begin
        m = MultiKernel([EQ() 0; 0 EQ()])
        gp = GP(m)
        x = collect(1.:5.)
        y = rand(5, 2)
        @test size(gp.m(x)) == (5, 2)
        pos = condition(gp, x, y)
        xp = collect(1.:4.)
        d = pos(xp)
        @test size(d.μ) == (4, 2)
        @test size(d.Σ) == (8, 8)

        mvn = MvNormal(gp, x)
        @test isa(mvn, MvNormal)
        @test size(mvn.μ) == (10,)
        @test size(mvn.Σ) == (10, 10)
    end

    @testset "Extended space" begin
        gp = GP(5 * ConstantMean(), NoiseKernel(EQ(), 11 * DiagonalKernel()))
        x = collect(1:3)
        @test gp(x).μ == fill(5.0, 6)
        @test diag(gp(x).Σ) ≈ [12.0, 1.0, 12.0, 1.0, 12.0, 1.0] atol = _ATOL_
        @test gp(Observed(x)).μ == fill(5.0, 3)
        @test diag(gp(Observed(x)).Σ) == [12.0, 12.0, 12.0]
        @test gp(Latent(x)).μ == fill(5.0, 3)
        @test diag(gp(Latent(x)).Σ) == [1.0, 1.0, 1.0]
        @test gp([Latent([1,2]), Observed([3]), Latent([4])]).μ == fill(5.0, 4)
        @test diag(gp([Latent([1,2]), Observed([3]), Latent([4])]).Σ) ≈
            [1., 1., 12., 1.] atol = _ATOL_
        @test all(
            credible_interval(gp, x)[2] .<
            credible_interval(gp, x)[1] .<
            credible_interval(gp, x)[3]
        )
        @test all(
            credible_interval(gp, Observed(x))[2] .<
            credible_interval(gp, Observed(x))[1] .<
            credible_interval(gp, Observed(x))[3]
        )
        @test all(
            credible_interval(gp, [Latent([1,2]), Observed([3]), Latent([4])])[2] .<
            credible_interval(gp, [Latent([1,2]), Observed([3]), Latent([4])])[1] .<
            credible_interval(gp, [Latent([1,2]), Observed([3]), Latent([4])])[3]
        )
        mvn = MvNormal(gp, x)
        @test isa(mvn, MvNormal)
        @test size(mvn.μ) == (6,)
        @test size(mvn.Σ) == (6, 6)
        mvn = MvNormal(gp, Observed(x))
        @test isa(mvn, MvNormal)
        @test size(mvn.μ) == (3,)
        @test size(mvn.Σ) == (3, 3)

        gp = GP(MultiMean(fill(ConstantMean(), 2)), NoiseKernel(EQ(), 11 * DiagonalKernel()))
        @test size(gp(x).μ) == (3, 2)
        @test all(
            credible_interval(gp, [1,5,50])[2] .<
            credible_interval(gp, [1,5,50])[1] .<
            credible_interval(gp, [1,5,50])[3]
        )
    end
end

@testset "Full LMM" begin
    xs = hcat([sin.(collect(1:6)./i) for i in 1:3]...)
    H = [1 2 3; 4 3 2; 4 4 4; 1 3 2; 7 6 3]
    y = (H * xs')'
    gp = GP(LMMKernel(3, 5, 1e-2, H, EQ()))
    x = collect(1.0:6.0)
    ngp = condition(gp, x, y)
    xx = collect(3.0:6.0)
    @test ngp.m(x) ≈ y atol=1e-2
    @test ngp.m(xx) ≈ ngp.m(x)[3:end,:] atol = _ATOL_
    @test ngp.k(x) ≈ ngp.k(x, x) atol = _ATOL_
    @test ngp.k(x, xx) ≈ ngp.k(xx, x)' atol = _ATOL_
    @test diag(ngp.k(xx)) ≈ diag(ngp.k(xx, true))  atol = 30 * _ATOL_ # Increasing
    # tolerance here because `ngp.k(xx, true)` calls `var`, which has further approximations.
    @test reshape(diag(ngp.k(xx)), 5, 4)' ≈ var(ngp.k, xx) atol = 30 * _ATOL_
    @test ngp.k(xx)[1:5, 1:5] ≈ hourly_cov(ngp.k, xx)[1:5, 1:5] atol = _ATOL_
    @test ngp.k(xx)[6:10, 6:10] ≈ hourly_cov(ngp.k, xx)[6:10, 6:10] atol = _ATOL_
    @test all(
        credible_interval(ngp, [1.,5.,50.])[2] .<
        credible_interval(ngp, [1.,5.,50.])[1] .<
        credible_interval(ngp, [1.,5.,50.])[3]
    )
    ngp2 = condition(gp, x[1:3], y[1:3, :])
    ngp2 = condition(ngp2, x[4:6], y[4:6, :])
    xxx = collect(5:9)
    @test ngp.m(xxx) ≈ ngp2.m(xxx) atol = _ATOL_
    @test ngp.k(xxx) ≈ ngp2.k(xxx) atol = _ATOL_

    gp = GP(LMMKernel(3, 5, fill(1e-2, 5), H, EQ()))
    x = collect(1.0:6.0)
    ngp = condition(gp, x, y)
    @test ngp.m(x) ≈ y atol=1e-2
    @test ngp.m(xx) ≈ ngp.m(x)[3:end,:] atol = _ATOL_
    @test ngp.k(x) ≈ ngp.k(x, x) atol = _ATOL_
    @test ngp.k(x, xx) ≈ ngp.k(xx, x)' atol = _ATOL_
    @test diag(ngp.k(xx)) ≈ diag(ngp.k(xx, true))  atol = 30 * _ATOL_ # Increasing
    # tolerance here because `ngp.k(xx, true)` calls `var`, which has further approximations.
    @test reshape(diag(ngp.k(xx)), 5, 4)' ≈ var(ngp.k, xx) atol = 30 * _ATOL_
end

@testset "Full OLMM" begin
    xs = hcat([sin.(2π*collect(0:0.1:2)./i) for i in 1:3]...)
    A = ones(5,5) + 2eye(5)
    U, S, V = svd(A)
    H = U * diagm(S)[:, 1:3]
    y = (H * xs')'

    # check optimised versions
    sgp = GP(OLMMKernel(3, 5, 1e-2, 1e-2, H, EQ()))
    gp = GP(OLMMKernel(3, 5, 1e-2, 1e-2, H, [EQ() for i in 1:3]))
    x = collect(0:0.1:2)
    @test mean(sgp(x)) ≈ mean(gp(x)) atol = _ATOL_
    @test cov(sgp(x)) ≈ cov(gp(x)) atol = _ATOL_
    @test var(gp(x)) ≈ var(sgp(x)) atol = _ATOL_

    sgp.k.S_sqrt = ones(3)
    gp.k.S_sqrt = ones(3)
    @test logpdf(gp, x, y) ≈ logpdf(sgp, x, y) atol = _ATOL_
    pgp = condition(gp, x, y)
    psgp = condition(sgp, x, y)
    @test mean(psgp(x)) ≈ mean(pgp(x)) atol = _ATOL_
    @test cov(psgp(x)) ≈ cov(pgp(x)) atol = _ATOL_
    @test var(pgp(x)) ≈ var(psgp(x)) atol = _ATOL_

    gp = GP(OLMMKernel(3, 5, 1e-2, 1e-2, H, [periodicise(EQ(), i) for i in 1:3]))
    x = collect(0:0.1:2)
    ngp = condition(gp, x, y)
    xx = collect(0:0.1:3)
    @test ngp.m(x) ≈ y atol=1e-1
    @test ngp.m(xx)[1:21, :] ≈ ngp.m(x) atol = _ATOL_
    P = GPForecasting.unwrap(gp.k.P)
    @test (P * ngp.m(x)')' ≈ xs atol = 1e-1
    # TODO: Implement k(x, y)
    # @test ngp.k(x) ≈ ngp.k(x, x) atol = _ATOL_
    # @test ngp.k(x, xx) ≈ ngp.k(xx, x)' atol = _ATOL_
    # @test diag(ngp.k(xx)) ≈ diag(ngp.k(xx, true))  atol = _ATOL_
    @test reshape(diag(ngp.k(xx)), 5, 31)' ≈ var(ngp.k, xx) atol = _ATOL_
    @test ngp.k(xx)[1:5, 1:5] ≈ hourly_cov(ngp.k, xx)[1:5, 1:5] atol = _ATOL_
    @test ngp.k(xx)[6:10, 6:10] ≈ hourly_cov(ngp.k, xx)[6:10, 6:10] atol = _ATOL_
    @test all(
        credible_interval(ngp, [1.,5.,50.])[2] .<
        credible_interval(ngp, [1.,5.,50.])[1] .<
        credible_interval(ngp, [1.,5.,50.])[3]
    )
    # TODO: work on reconditioning
    # ngp2 = condition(gp, x[1:3], y[1:3, :])
    # ngp2 = condition(ngp2, x[4:6], y[4:6, :])
    # xxx = collect(5:9)
    # @test ngp.m(xxx) ≈ ngp2.m(xxx) atol = _ATOL_
    # @test ngp.k(xxx) ≈ ngp2.k(xxx) atol = _ATOL_

    gp2 = GP(
        OLMMKernel(3, 5, 1e-2, 1e-2, U[:, 1:3], S[1:3], [periodicise(EQ(), i) for i in 1:3])
    )
    ngp2 = condition(gp2, x, y)
    @test gp.k(x) ≈ gp2.k(x) atol = _ATOL_
    @test ngp.k(x) ≈ ngp2.k(x) atol = _ATOL_
    @test gp.m(x) ≈ gp2.m(x) atol = _ATOL_
    @test ngp.m(x) ≈ ngp2.m(x) atol = _ATOL_

    gp = GP(OLMMKernel(3, 5, 1e-2, fill(1e-2, 3), H, [periodicise(EQ(), i) for i in 1:3]))
    x = collect(0:0.1:2)
    ngp = condition(gp, x, y)
    @test ngp.m(x) ≈ y atol=1e-1
    @test ngp.m(xx)[1:21, :] ≈ ngp.m(x) atol = _ATOL_
    # TODO: Implement k(x, y)
    # @test ngp.k(x) ≈ ngp.k(x, x) atol = _ATOL_
    # @test ngp.k(x, xx) ≈ ngp.k(xx, x)' atol = _ATOL_
    # @test diag(ngp.k(xx)) ≈ diag(ngp.k(xx, true))  atol = _ATOL_
    @test reshape(diag(ngp.k(xx)), 5, 31)' ≈ var(ngp.k, xx) atol = _ATOL_
    Ug = GPForecasting.greedy_U(gp.k, x, y)
    @test sum(Ug' * Ug) ≈ 3.0 atol = _ATOL_
end
