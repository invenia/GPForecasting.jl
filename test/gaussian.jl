@testset "Gaussian" begin
    n = Gaussian([1.,2e10,3.], Eye(3))
    samp = sample(n)
    @test size(samp) == (3,)
    samp = sample(n, 3)
    @test size(samp) == (3, 3)
    @test all(samp[1, :] .< samp[2, :])
    @test isa(logpdf(n, [1.0, 2.0, 3.0]), Real)
    @test isa(logpdf(n, [[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]), Real)
    @test_throws DimensionMismatch logpdf(n, [1., 2., 3., 4.])

    mvg = Gaussian(Float64[1 2; 3 4; 5 6], Eye(6))
    mvn = MvNormal(mvg)
    @test isa(mvn, MvNormal)
    @test size(mvn.μ) == (6,)
    @test size(mvn.Σ) == (6, 6)
    @test mvn.μ == [1, 2, 3, 4, 5, 6]

    a = rand(6)
    mvg = Gaussian(Float64[1 -2e10; 3 -4e10; 5 -6e10], a * a' + 1e-3 * Eye(6))
    samp = sample(mvg)
    @test all(samp[:, 1] .> samp[:, 2])
    samp = sample(mvg, 4)
    @test size(samp) == (3, 2, 4)
    @test all(samp[:, 1, :] .> samp[:, 2, :])

    g = Gaussian(ones(3, 2), Eye(6))
    @test g.chol === nothing
    @test cholesky(g).U ≈ UpperTriangular(Eye(6)) atol = _ATOL_ rtol = _RTOL_
    @test mean(g) ≈ ones(3, 2) atol = _ATOL_ rtol = _RTOL_
    @test cov(g) ≈ Eye(6) atol = _ATOL_ rtol = _RTOL_
    @test var(g) ≈ ones(3, 2) atol = _ATOL_ rtol = _RTOL_

    g = Gaussian(zeros(4, 5), BlockDiagonal(fill(diagm(0 => collect(1:5)), 4)))
    @test var(g) ≈ [1 2 3 4 5; 1 2 3 4 5; 1 2 3 4 5; 1 2 3 4 5] atol = _ATOL_ rtol = _RTOL_

    g = Gaussian(ones(5, 2), BlockDiagonal(AbstractMatrix{Float64}[Eye(2) for i in 1:5]));
    @test g.chol === nothing
    @test cholesky(g).U ≈ BlockDiagonal([cholesky(Eye(2)).U for i in 1:5]) atol = _ATOL_ rtol = _RTOL_
    @test mean(g) ≈ ones(5, 2) atol = _ATOL_ rtol = _RTOL_
    @test cov(g) ≈ BlockDiagonal([Eye(2) for i in 1:5]) atol = _ATOL_ rtol = _RTOL_
    @test var(g) ≈ ones(5, 2) atol = _ATOL_ rtol = _RTOL_
    @test size(rand(g, 5)) == (5, 2, 5)

    mu = zeros(3, 5)
    Σ = BlockDiagonal(fill(Eye(5), 3))
    diagonal_zeromean_gaussian1 = Gaussian(mu, Σ, cholesky(Σ))
    diagonal_zeromean_gaussian2 = Gaussian(mu, BlockDiagonal(fill(Eye(5), 3)))
    diagonal_zeromean_gaussian3 = Gaussian(mu, Eye(15))
    # TODO: change the logpdf function so it detects a weird case like g4. It isn't something
    # we'll normally be using, though
    # g4 = Gaussian(mu, BlockDiagonal([Eye(10), Eye(5)]))
    diagonal_zeromean_gaussian5 = Gaussian(Zeros(3, 5), Eye(15))
    r = rand(15)
    nondiagonal_gaussian = Gaussian(Zeros(3, 5), Eye(15) + r *r')
    multivariate_gaussian = Gaussian(rand(3), Eye(3))
    xs = [rand(3, 5) for i in 1:3]
    x = xs[1]

    for g in (
        diagonal_zeromean_gaussian2,
        diagonal_zeromean_gaussian3,
        diagonal_zeromean_gaussian5
    )
        for f in (
            Distributions.loglikelihood,
            Metrics.marginal_gaussian_loglikelihood,
            Metrics.joint_gaussian_loglikelihood,
        )
            @test f(diagonal_zeromean_gaussian1, xs) ≈ f(g, xs) atol = _ATOL_ rtol = _RTOL_
        end
        # Values here should coincide because none of the Gaussians have non-zero entries
        # outside the diagonal of the covariance.
        @test marginal_mean_logloss(diagonal_zeromean_gaussian1, x) ≈ marginal_mean_logloss(g, x) atol = _ATOL_ rtol = _RTOL_
        @test joint_mean_logloss(g, x) ≈ marginal_mean_logloss(g, x) atol = _ATOL_ rtol = _RTOL_
    end
    @test joint_gaussian_loglikelihood(nondiagonal_gaussian, xs) ==
        Distributions.loglikelihood(nondiagonal_gaussian, xs) !=
        marginal_gaussian_loglikelihood(nondiagonal_gaussian, xs)
    for f in (
        Distributions.loglikelihood,
        Metrics.marginal_gaussian_loglikelihood,
        Metrics.joint_gaussian_loglikelihood,
    )
        x = rand(3, 5)
        @test f(multivariate_gaussian, x) ≈ sum(
            broadcast(s -> logpdf(multivariate_gaussian, s), [x[:, i] for i in 1:5])
        ) atol = _ATOL_ rtol = _RTOL_
    end

    # Test Adjoint constructors
    g = Gaussian(rand(3, 2), rand(6, 6)')
    @test isa(mean(g), Matrix)
    @test isa(cov(g), Matrix)
    g = Gaussian(rand(3, 2)', rand(6, 6))
    @test isa(mean(g), Matrix)
    @test isa(cov(g), Matrix)
    g = Gaussian(rand(3, 2)', rand(6, 6)')
    @test isa(mean(g), Matrix)
    @test isa(cov(g), Matrix)
    Σ = (Y->Y'Y)(rand(6, 6))'  # doing Y'Y here ensures that cholesky won't fail
    g = Gaussian(rand(3, 2), Σ, cholesky(copy(Σ)))
    @test isa(mean(g), Matrix)
    @test isa(cov(g), Matrix)
    Σ = (Y->Y'Y)(rand(6, 6))
    g = Gaussian(rand(3, 2)', Σ, cholesky(Σ))
    @test isa(mean(g), Matrix)
    @test isa(cov(g), Matrix)
    Σ = (Y->Y'Y)(rand(6, 6))'
    g = Gaussian(rand(3, 2)', Σ, cholesky(copy(Σ)))
    @test isa(mean(g), Matrix)
    @test isa(cov(g), Matrix)

    @testset "show" begin
        g = Gaussian(Float64[1 2; 3 4; 5 6], Eye(6))
        @test sprint(show, g) == """
            Gaussian{Array{Float64,2}, $(typeof(Eye(6)))}(
                μ: [1.0 2.0; 3.0 4.0; 5.0 6.0]
                Σ: 6×6 Eye{Float64}
                chol: <not yet computed>
            )"""
        c = cholesky(g)
        context = IOContext(IOBuffer(), :compact=>true, :limit=>true)
        @test sprint(show, g) == """
            Gaussian{Array{Float64,2}, $(typeof(Eye(6)))}(
                μ: [1.0 2.0; 3.0 4.0; 5.0 6.0]
                Σ: 6×6 Eye{Float64}
                chol: $(sprint(show, c, context=context))
            )"""
    end
end
