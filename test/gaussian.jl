using ModelAnalysis

@testset "Gaussian" begin
    n = Gaussian([1.,2.,3.], eye(3))

    @test size(sample(n, 3)) == (3, 3)
    @test isa(logpdf(n, [1., 2., 3.]), Real)
    @test_throws DimensionMismatch logpdf(n, [1., 2., 3., 4.])

    mvg = Gaussian([1 2; 3 4; 5 6], eye(6))
    mvn = MvNormal(mvg)
    @test isa(mvn, MvNormal)
    @test size(mvn.μ) == (6,)
    @test size(mvn.Σ) == (6, 6)
    @test mvn.μ == [1, 2, 3, 4, 5, 6]

    g = Gaussian(ones(3, 2), eye(6))
    @test g.U == Matrix(0, 0)
    @test chol(g) ≈ UpperTriangular(eye(6)) atol = _ATOL_
    @test mean(g) ≈ ones(3, 2) atol = _ATOL_
    @test cov(g) ≈ eye(6) atol = _ATOL_
    @test var(g) ≈ ones(3, 2) atol = _ATOL_

    g = Gaussian(zeros(4, 5), BlockDiagonal(fill(diagm(collect(1:5)), 4)))
    @test var(g) ≈ [1 2 3 4 5; 1 2 3 4 5; 1 2 3 4 5; 1 2 3 4 5] atol = _ATOL_

    g = Gaussian(ones(5, 2), BlockDiagonal(AbstractMatrix{Float64}[eye(2) for i in 1:5]));
    @test g.U == Matrix(0, 0)
    @test chol(g) ≈ BlockDiagonal([chol(eye(2)) for i in 1:5]) atol = _ATOL_
    @test mean(g) ≈ ones(5, 2) atol = _ATOL_
    @test cov(g) ≈ BlockDiagonal([eye(2) for i in 1:5]) atol = _ATOL_
    @test var(g) ≈ ones(5, 2) atol = _ATOL_
    @test size(rand(g, 5)) == (5, 2, 5)

    g1 = Gaussian(zeros(3, 5), BlockDiagonal(fill(eye(5), 3)), BlockDiagonal(fill(eye(5), 3)))
    g2 = Gaussian(zeros(3, 5), BlockDiagonal(fill(eye(5), 3)))
    g3 = Gaussian(zeros(3, 5), eye(15))
    g4 = Gaussian(zeros(3, 5), BlockDiagonal([eye(10), eye(5)]))
    @test mll_joint(g1, ones(3, 5)) ≈ mll_joint(g2, ones(3, 5)) atol = _ATOL_
    @test mll_joint(g1, ones(3, 5)) ≈ mll_joint(g3, ones(3, 5)) atol = _ATOL_
    @test mll_joint(g1, ones(3, 5)) ≈ mll_joint(g4, ones(3, 5)) atol = _ATOL_
end
