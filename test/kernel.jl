@testset "Kernel" begin

    @testset "Each Kernel" begin
        for k in [
            EQ(),
            ConstantKernel(),
            RQ(1.5),
            MA(1/2),
            MA(3/2),
            MA(5/2),
            SimilarHourKernel(3, [3, 2, 1]),
        ]
            @test (0.0 * k)([5.]) ≈ [0.0] atol = _ATOL_
            @test k([5., 6.]) ≈ k([5., 6.], [5., 6.]) atol = _ATOL_
            @test diag(k([1., 2., 3.])) ≈ var(k, [1., 2., 3.]) atol = _ATOL_
            @test hourly_cov(k, [1., 2., 3.]) ≈ diagm(var(k, [1., 2., 3.])) atol = _ATOL_
            @test !isMulti(k)
        end
        @test_throws ArgumentError MA(6)([4.])

        @testset "BinaryKernel" begin
            k = BinaryKernel(5, 1, 6)
            @test GPForecasting.unwrap(set(k, k[:]).Θ₃) ≈ sqrt(5) atol = _ATOL_
            k.Θ₃.p = -600
            k.Θ₁.p = 500
            @test GPForecasting.unwrap(set(k, k[:]).Θ₃) ≈ -sqrt(500) atol = _ATOL_
            k.Θ₃.p = 6
            k.Θ₁.p = 500
            @test GPForecasting.unwrap(set(k, k[:]).Θ₃) ≈ 6 atol = _ATOL_
            @test k([1, 0, 1]) ≈ [500 6 500; 6 1 6; 500 6 500]
            @test !isMulti(k)
        end

        @test_throws ArgumentError SimilarHourKernel(30, zeros(30))
        @test_throws ArgumentError SimilarHourKernel(0, zeros(0))
        @test_throws DimensionMismatch SimilarHourKernel(5, zeros(3))
        K = SimilarHourKernel(3, [3, 2, 1])(collect(1:24), 50*24 .+ collect(25:48))
        @test diag(K) == fill(3, 24)
        @test diag(K, 1) == fill(2, 23)
        @test diag(K, 2) == fill(1, 22)
        @test diag(K, 3) == fill(0, 21)
        @test ZeroKernel()(rand(5, 3)) ≈ zeros(5, 5)
        @test !isMulti(ZeroKernel())
    end

    @testset "Algebra" begin
        k = 5 * EQ()
        @test GPForecasting.unwrap((6 * k).scale) == 30
        kk = 0 * EQ()
        @test kk + k == k + kk == k
        @test isa(k * kk, ZeroKernel)
        @test isa(kk * k, ZeroKernel)
        kkk = EQ()
        @test kkk + kk == kk + kkk == kkk
        @test isa(kkk * kk, ZeroKernel)
        @test isa(kk * kkk, ZeroKernel)
    end

    @testset "Parameter" begin
        k = EQ() ▷ 0.7
        @test k[:][1] ≈ -0.35667637251118145 atol = _ATOL_
        k = EQ() ▷ Named(0.7, "named")
        @test k[:][1] ≈ -0.35667637251118145 atol = _ATOL_
        k = EQ() ▷ Positive(0.7)
        @test k[:][1] ≈ -0.35667637251118145 atol = _ATOL_
        k = EQ() ▷ Bounded(0.7, 0.1, 1.0)
        @test k[:][1] ≈ 0.6931455138937418 atol = _ATOL_
        k = EQ() ▷ [1.0, 2.0, 3.0]
        @test k[:] ≈ [-1.0e-6, 0.6931466805598203, 1.0986119553347207] atol = _ATOL_
        @test k([1. 2. 3.], [4. 5. 6.])[1] ≈ 0.002187491118182885 atol = _ATOL_
    end

    @testset "Periodic" begin
        for k in [
            EQ(),
            ConstantKernel(),
            RQ(Fixed(1.5)),
            MA(1/2),
            MA(3/2),
            MA(5/2),
        ]
            k = periodicise(k, 10)
            @test k([1.,2.]) ≈ k([11.,12.]) atol = _ATOL_
            k = periodicise(k ▷ [2.0, 3.0], [2π, 3π])
            @test k[:][1:2] ≈ [1.8378769072543897, 2.2433420684142087] atol = _ATOL_
            @test k[:][3:4] ≈ [0.6931466805598203, 1.0986119553347207] atol = _ATOL_
        end

        x = collect(1.:0.1:5)
        @test periodicise(EQ(), 2.3)(x) ≈ exp.(-2sin.(π .* (x .- x') ./ 2.3).^2) # see
        # Rasmussen pg 92
    end

    @testset "Specified Quantity" begin
        k = EQ()
        df = DataFrame([[1.,2.,3.], [1.,1.,1.]], [:input1, :input2])
        sqk1 = k ← :input1
        sqk2 = k ← :input2
        @test !isMulti(sqk1)
        @test !(sqk1(df) ≈ sqk2(df))
        @test sqk2(df) ≈ ones(3, 3) atol = _ATOL_
    end

    @testset "Sum and Products" begin
        k_sum = EQ() + EQ()
        @test !isMulti(k_sum)
        k_prod = (2 * EQ()) * (3 * EQ())
        @test !isMulti(k_prod)
        @test k_sum([1.])[1, 1] ≈ 2.0 atol = _ATOL_
        @test k_sum([1.], [2.])[1, 1] ≈ 1.2130613194252668 atol = _ATOL_
        @test k_prod([1.], [1.])[1, 1] ≈ 6.0 atol = _ATOL_
        @test k_prod([1.], [2.])[1, 1] ≈ 6 * 0.36787944117144233 atol = _ATOL_
    end

    @testset "Set and Get" begin
        k = 5. * EQ() ▷ Fixed(2.) + Named(Fixed(2.), "variance") * EQ() ▷ Named(3., "scale")

        @test k[:] ≈ [1.6094377124340804, 1.0986119553347207] atol = _ATOL_
        @test set(k, k[:] .+ log(2.))[:] ≈ [2.302584892994026, 1.791759135894666] atol = _ATOL_
        @test k["variance"] ≈ 2.0 atol = _ATOL_
        @test set(k, "variance" => 5.)["variance"] ≈ 5.0 atol = _ATOL_
    end

    @testset "PosteriorKernel" begin
        pk = PosteriorKernel(ConstantKernel(), [1,2,3], eye(3))
        @test !isMulti(pk)
        @test pk([1,2], [1,2]) ≈ [-2. -2.; -2. -2.] atol = _ATOL_
    end

    @testset "Noise and Multi-output kernels" begin
        m = MultiKernel([EQ() 2*EQ(); EQ()+2 EQ()▷3])

        x = [1., 2., 3., 11., 22.]
        y = [1., 2., 3., 4., 5.]
        @test isMulti(m)
        @test m(x, 1, 1) ≈ EQ()(x) atol = _ATOL_
        @test (EQ() + m)(x) ≈ MultiKernel(
            [EQ()+EQ() 2*EQ()+EQ(); EQ()+2+EQ() (EQ()▷3)+EQ()]
        )(x) atol = _ATOL_
        @test (EQ() + m)(x) ≈ (1 * EQ() + m)(x) atol = _ATOL_
        @test ((m + m) + m)(x) ≈ 3 .* m(x) atol = _ATOL_
        @test diag(hourly_cov(m, x)) ≈ diag(m(x)) atol = _ATOL_
        @test hourly_cov(m, x)[1:2, 1:2] ≈ m(x)[1:2, 1:2] atol = _ATOL_
        @test !isapprox(hourly_cov(m, x)[2:4, 1:2], m(x)[2:4, 1:2], atol = _ATOL_)

        ox = Observed(x)
        oy = Observed(y)
        py = Latent(y)
        px = Latent(x)
        mx = [Latent([1,2,3]), Observed([11, 22]), Latent([15])]

        nk = GPForecasting.NoiseKernel(EQ(), 12*DiagonalKernel()) # Giant noise to make it easy to spot
        @test !isMulti(nk)
        @test nk(px, oy) ≈ nk(ox, py) ≈ nk(px, py)
        @test nk(ox, oy) ≈ nk(px, py) + [12 0 0 0 0; 0 12 0 0 0; 0 0 12 0 0; 0 0 0 0 0; 0 0 0 0 0] atol = _ATOL_
        @test size(nk(ox, oy)) == (5, 5)
        @test diag(nk(mx)) ≈ [1.0, 1.0, 1.0, 13.0, 13.0, 1.0] atol = _ATOL_
        @test nk(x, y)[1:2, 1:2] ≈ [13.0 1.0; 1.0 1.0] atol = _ATOL_
        @test diag(hourly_cov(nk, ox)) ≈ diag(nk(ox)) atol = _ATOL_
        @test diag(hourly_cov(nk, px)) ≈ diag(nk(px)) atol = _ATOL_
        @test diag(hourly_cov(nk, x)) ≈ diag(nk(x)) atol = _ATOL_
        @test hourly_cov(nk, x)[1:2, 1:2] ≈ nk(x)[1:2, 1:2] atol = _ATOL_
        @test !isapprox(hourly_cov(nk, x)[2:4, 1:2], nk(x)[2:4, 1:2], atol = _ATOL_)
        @test diag(hourly_cov(nk, mx)) ≈ diag(nk(mx)) atol = _ATOL_

        nk = GPForecasting.NoiseKernel(m, 12*DiagonalKernel())
        @test isMulti(nk)
        @test nk(px, oy) ≈ nk(ox, py) ≈ nk(px, py)
        @test nk(ox, oy)[1:4, 1:2] ≈ (
            nk(px, py)[1:4, 1:2] .+ float.([12 12; 12 12; 0 0; 0 0])
        ) atol = _ATOL_
        @test size(nk(ox, oy)) == (10, 10)
        @test diag(hourly_cov(nk, ox)) ≈ diag(nk(ox)) atol = _ATOL_
        @test hourly_cov(nk, ox)[1:2, 1:2] ≈ nk(ox)[1:2, 1:2] atol = _ATOL_
        @test !isapprox(hourly_cov(nk, ox)[2:4, 1:2], nk(ox)[2:4, 1:2], atol = _ATOL_)
        @test diag(hourly_cov(nk, px)) ≈ diag(nk(px)) atol = _ATOL_
        @test hourly_cov(nk, px)[1:2, 1:2] ≈ nk(px)[1:2, 1:2] atol = _ATOL_
        @test !isapprox(hourly_cov(nk, px)[2:4, 1:2], nk(px)[2:4, 1:2], atol = _ATOL_)
        @test diag(hourly_cov(nk, x)) ≈ diag(nk(x)) atol = _ATOL_
        @test hourly_cov(nk, x)[1:4, 1:4] ≈ nk(x)[1:4, 1:4] atol = _ATOL_
        @test !isapprox(hourly_cov(nk, x)[5:8, 1:4], nk(x)[5:8, 1:4], atol = _ATOL_)
        @test diag(hourly_cov(nk, mx)) ≈ diag(nk(mx)) atol = _ATOL_

        nk = GPForecasting.NoiseKernel(m, MultiKernel([12*DiagonalKernel() 0; 0 35*DiagonalKernel()]))
        @test isMulti(nk)
        @test nk(px, oy) ≈ nk(ox, py) ≈ nk(px, py)
        @test nk(ox, oy)[3:6, 3:4] ≈ (
            nk(px, py)[3:6, 3:4] .+ float.([12 0; 0 35; 0 0; 0 0])
        ) atol = _ATOL_
        @test nk(ox, oy)[5:6, 5:6] ≈ (nk(px, py)[5:6, 5:6] .+ float.([12 0; 0 35])) atol = _ATOL_
        @test nk(ox, oy)[6:10, 1:5] ≈ nk(px, py)[6:10, 1:5] atol = _ATOL_
        @test nk(ox, oy)[1:5, 6:10] ≈ nk(px, py)[1:5, 6:10] atol = _ATOL_
        @test size(nk(ox, oy)) == (10, 10)
        @test diag(hourly_cov(nk, ox)) ≈ diag(nk(ox)) atol = _ATOL_
        @test hourly_cov(nk, ox)[1:2, 1:2] ≈ nk(ox)[1:2, 1:2] atol = _ATOL_
        @test !isapprox(hourly_cov(nk, ox)[2:4, 1:2], nk(ox)[2:4, 1:2], atol = _ATOL_)
        @test diag(hourly_cov(nk, px)) ≈ diag(nk(px)) atol = _ATOL_
        @test hourly_cov(nk, px)[1:2, 1:2] ≈ nk(px)[1:2, 1:2] atol = _ATOL_
        @test !isapprox(hourly_cov(nk, px)[2:4, 1:2], nk(px)[2:4, 1:2], atol = _ATOL_)
        @test diag(hourly_cov(nk, x)) ≈ diag(nk(x)) atol = _ATOL_
        @test hourly_cov(nk, x)[1:4, 1:4] ≈ nk(x)[1:4, 1:4] atol = _ATOL_
        @test !isapprox(hourly_cov(nk, x)[5:8, 1:4], nk(x)[5:8, 1:4], atol = _ATOL_)
        @test diag(hourly_cov(nk, mx)) ≈ diag(nk(mx)) atol = _ATOL_
    end

    @testset "Matrix algebra" begin
        H = [3 2; 4 1; 2 5]
        @test isa(H * EQ(), MultiKernel)
        @test (H * EQ())(collect(1.:3.)) ≈ (EQ() * H)(collect(1.:3.)) atol = _ATOL_
        @test H * ZeroKernel() == ZeroKernel() * H == Matrix{Kernel}([0 0; 0 0; 0 0])
        mk = MultiKernel([EQ() 3 * EQ(); 0 5 * EQ()])
        @test isa(H * (3 * mk), ScaledKernel)
        @test isMulti(H * (3 * mk))
        @test_throws DimensionMismatch (3 * mk) * H
        @test isMulti(H * (EQ() + RQ(5.5)))
        @test isMulti((EQ() + RQ(5.5)) * H)
        k = periodicise(EQ(), 2.)
        @test (H * k)([1., 2.]) ≈ (k * H)([3., 4.]) atol = _ATOL_
    end

    @testset "Naive LMM" begin
        H = [1 2 3; 4 3 2; 4 4 4; 1 3 2; 7 6 3]
        nlmm = NaiveLMMKernel(3, 1e-2, H, EQ())
        vnlmm = verynaiveLMMKernel(3, 5, 1e-2, H, EQ())
        @test isa(nlmm, NaiveLMMKernel)
        @test isa(vnlmm, MultiKernel)
        x = collect(1:5)
        @test nlmm(x) ≈ vnlmm(x) atol = _ATOL_
        @test isMulti(nlmm)
    end
end