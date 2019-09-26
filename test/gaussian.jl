@testset "Gaussian" begin
    n = Gaussian([1.,2e10,3.], Eye(3))
    samp = sample(n)
    @test size(samp) == (3,)
    samp = sample(n, 3)
    @test size(samp) == (3, 3)
    @test all(samp[1, :] .< samp[2, :])
    @test isa(logpdf(n, [1., 2., 3.]), Real)
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
    @test cholesky(g).U ≈ UpperTriangular(Eye(6)) atol = _ATOL_
    @test mean(g) ≈ ones(3, 2) atol = _ATOL_
    @test cov(g) ≈ Eye(6) atol = _ATOL_
    @test var(g) ≈ ones(3, 2) atol = _ATOL_

    g = Gaussian(zeros(4, 5), BlockDiagonal(fill(diagm(0 => collect(1:5)), 4)))
    @test var(g) ≈ [1 2 3 4 5; 1 2 3 4 5; 1 2 3 4 5; 1 2 3 4 5] atol = _ATOL_

    g = Gaussian(ones(5, 2), BlockDiagonal(AbstractMatrix{Float64}[Eye(2) for i in 1:5]));
    @test g.chol === nothing
    @test cholesky(g).U ≈ BlockDiagonal([cholesky(Eye(2)).U for i in 1:5]) atol = _ATOL_
    @test mean(g) ≈ ones(5, 2) atol = _ATOL_
    @test cov(g) ≈ BlockDiagonal([Eye(2) for i in 1:5]) atol = _ATOL_
    @test var(g) ≈ ones(5, 2) atol = _ATOL_
    @test size(rand(g, 5)) == (5, 2, 5)

    mu = zeros(3, 5)
    Σ = BlockDiagonal(fill(Eye(5), 3))
    g1 = Gaussian(mu, Σ, cholesky(Σ))
    g2 = Gaussian(mu, BlockDiagonal(fill(Eye(5), 3)))
    g3 = Gaussian(mu, Eye(15))
    g4 = Gaussian(mu, BlockDiagonal([Eye(10), Eye(5)]))
    g5 = Gaussian(Zeros(3, 5), Eye(15))
    # TODO: tests of loglikelihood based functions

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
                Σ: [1.0 0.0 … 0.0 0.0; 0.0 1.0 … 0.0 0.0; … ; 0.0 0.0 … 1.0 0.0; 0.0 0.0 … 0.0 1.0]
                chol: <not yet computed>
            )"""
        c = cholesky(g)
        context = IOContext(IOBuffer(), :compact=>true, :limit=>true)
        @test sprint(show, g) == """
            Gaussian{Array{Float64,2}, $(typeof(Eye(6)))}(
                μ: [1.0 2.0; 3.0 4.0; 5.0 6.0]
                Σ: [1.0 0.0 … 0.0 0.0; 0.0 1.0 … 0.0 0.0; … ; 0.0 0.0 … 1.0 0.0; 0.0 0.0 … 0.0 1.0]
                chol: $(sprint(show, c, context=context))
            )"""
    end
end
