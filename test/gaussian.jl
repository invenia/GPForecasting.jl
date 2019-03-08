using ModelAnalysis

@testset "Gaussian" begin
    n = Gaussian([1.,2.,3.], Eye(3))

    @test size(sample(n, 3)) == (3, 3)
    @test isa(logpdf(n, [1., 2., 3.]), Real)
    @test_throws DimensionMismatch logpdf(n, [1., 2., 3., 4.])

    mvg = Gaussian(Float64[1 2; 3 4; 5 6], Eye(6))
    mvn = MvNormal(mvg)
    @test isa(mvn, MvNormal)
    @test size(mvn.μ) == (6,)
    @test size(mvn.Σ) == (6, 6)
    @test mvn.μ == [1, 2, 3, 4, 5, 6]

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

    g1 = Gaussian(zeros(3, 5), BlockDiagonal(fill(Eye(5), 3)), BlockDiagonal(fill(Eye(5), 3)))
    g2 = Gaussian(zeros(3, 5), BlockDiagonal(fill(Eye(5), 3)))
    g3 = Gaussian(zeros(3, 5), Eye(15))
    g4 = Gaussian(zeros(3, 5), BlockDiagonal([Eye(10), Eye(5)]))
    @test mll_joint(g1, ones(3, 5)) ≈ mll_joint(g2, ones(3, 5)) atol = _ATOL_
    @test mll_joint(g1, ones(3, 5)) ≈ mll_joint(g3, ones(3, 5)) atol = _ATOL_
    @test mll_joint(g1, ones(3, 5)) ≈ mll_joint(g4, ones(3, 5)) atol = _ATOL_

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
    g = Gaussian(rand(3, 2), rand(6, 6)', rand(6, 6))
    @test isa(mean(g), Matrix)
    @test isa(cov(g), Matrix)
    g = Gaussian(rand(3, 2)', rand(6, 6), rand(6, 6))
    @test isa(mean(g), Matrix)
    @test isa(cov(g), Matrix)
    g = Gaussian(rand(3, 2)', rand(6, 6)', rand(6, 6))
    @test isa(mean(g), Matrix)
    @test isa(cov(g), Matrix)
end
