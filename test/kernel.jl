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
            DotKernel(),
            HazardKernel(),
            RootLog(),
            DiagonalKernel(),
            ZeroKernel(),
            periodicise(EQ(), 5),
            CosineKernel(),
            HeteroskedasticDiagonalKernel(10),
            NKN(
                [EQ(), 2EQ(), RQ(0.5)],
                (LinearLayer((4, 3)), ProductLayer((2, 4)), LinearLayer((1, 2)))
            ),
        ]
            @test (0.0 * k)([5.]) ≈ [0.0] atol = _ATOL_ rtol = _RTOL_
            @test k([5., 6.]) ≈ k([5., 6.], [5., 6.]) atol = _ATOL_ rtol = _RTOL_
            # Tolerance is high below because of the RootLog kernel. TODO: Implement a
            # more stable version of it.
            @test diag(k([1., 2., 3.])) ≈ var(k, [1., 2., 3.]) atol = _ATOL_ rtol = _RTOL_
            @test hourly_cov(k, [1., 2., 3.]) ≈ Diagonal(var(k, [1., 2., 3.])) atol = _ATOL_ rtol = _RTOL_
            @test !isMulti(k)
            @test isposdef(k([1., 2., 3.]) + GPForecasting._EPSILON_^2 * I)
            @test isa(k(1, 1), AbstractMatrix)
            @test isa(k([1], [1]), AbstractMatrix)
            @test isa(k(1, [1, 2]), AbstractMatrix)
            @test isa(k([1, 2], 1), AbstractMatrix)
            @test isa(sprint(show, k), String)
            @test GPForecasting.is_not_noisy(k)
            # Tolerance is high below because of the RootLog kernel. TODO: Implement a
            # more stable version of it.
            @test diag(k([1., 2., 3.])) ≈ elwise(k, [1., 2., 3.]) atol = _ATOL_ rtol = _RTOL_
            @test diag(k([1., 2., 3.], [3., 2., 5.])) ≈ elwise(k, [1., 2., 3.], [3., 2., 5.]) atol = _ATOL_ rtol = _RTOL_
            @test elwise(k, [1., 2., 3.]) ≈ elwise(k, [1., 2., 3.], [1., 2., 3.])
            @test isa(elwise(k, [1., 2., 3.]), AbstractVector)
            @test_throws Any elwise(k, [1., 2., 3.], [1., 2., 3., 4.])
        end
        @test_throws ArgumentError MA(6)([4.])

        @testset "NKN" begin
            k = NKN(
                [EQ(), 2EQ(), RQ(0.5)],
                (LinearLayer((4, 3)), ProductLayer((2, 4)), LinearLayer((1, 2)))
            )
            @test isa(GPForecasting.equivalent_kernel(k), Kernel)
            @test isa(logpdf(GP(k), [1., 2., 3.], [11., 23., 45.]), Real)
            x = [1, 2, 3]
            @test k(x, x) ≈ k(x) atol = _ATOL_
            @test GPForecasting.equivalent_kernel(k)(x) ≈ k(x) atol = _ATOL_
        end

        @testset "DotKernel" begin
            k = DotKernel()
            x = [1. 0. 0.; 1. 1. 0.; 0. 0.5 0.]
            y = [0.5 1. 1.; 1. 1. 1.]
            @test k(x) ≈ [1. 1. 0.; 1.0 2.0 0.5; 0. 0.5 0.25] atol = _ATOL_ rtol = _RTOL_
            @test k(x, y) ≈ [0.5 1.0; 1.5 2.0; 0.5 0.5] atol = _ATOL_ rtol = _RTOL_
            @test isposdef(k(x) + 1e-10 * I)
            @test GPForecasting.is_not_noisy(k)
            k = DotKernel(1.0)
            @test k(x) ≈ [2.0 1.0 1.5; 1.0 1.0 1.0; 1.5 1.0 2.25]
            k = DotKernel([1.0, 1.0, 0.0])
            @test k(x) ≈ [1.0 0.0 0.5; 0.0 0.0 0.0; 0.5 0.0 1.25]
            @test isposdef(k(x) + 1e-10 * I)
        end

        @testset "HazardKernel" begin
            k = HazardKernel()
            x = [1.0 0.5; 0.0 0.0]
            y = [0. 0.; 0. 0.; 0. 0.]
            @test k(x) ≈ [1.25 0.; 0. 1.] atol = _ATOL_ rtol = _RTOL_
            @test k(x, y) ≈ [0. 0. 0.; 1. 1. 1.] atol = _ATOL_ rtol = _RTOL_
            k = HazardKernel(0.2)
            @test k(x) ≈ [1.29 0.2; 0.2 1.] atol = _ATOL_ rtol = _RTOL_
            @test k(x, y) ≈ [0.2 0.2 0.2; 1. 1. 1.] atol = _ATOL_ rtol = _RTOL_
            k = HazardKernel(0.2, [5 2])
            @test k(x) ≈ [26.04 0.2; 0.2 1.] atol = _ATOL_ rtol = _RTOL_
            @test k(x, y) ≈ [0.2 0.2 0.2; 1. 1. 1.] atol = _ATOL_ rtol = _RTOL_
            @test GPForecasting.is_not_noisy(k)
        end

        @testset "BinaryKernel" begin
            k = BinaryKernel(5, 1, 6)
            @test GPForecasting.unwrap(set(k, k[:]).Θ₃) ≈ sqrt(5) atol = _ATOL_ rtol = _RTOL_
            k.Θ₃.p = -600
            k.Θ₁.p = 500
            @test GPForecasting.unwrap(set(k, k[:]).Θ₃) ≈ -sqrt(500) atol = _ATOL_ rtol = _RTOL_
            k.Θ₃.p = 6
            k.Θ₁.p = 500
            @test GPForecasting.unwrap(set(k, k[:]).Θ₃) ≈ 6 atol = _ATOL_ rtol = _RTOL_
            @test k([1, 0, 1]) ≈ [500 6 500; 6 1 6; 500 6 500]
            @test !isMulti(k)
            @test isa(sprint(show, k), String)
            @test GPForecasting.is_not_noisy(k)
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
        @test GPForecasting.is_not_noisy(k)
        kk = 0 * EQ()
        sk = ScaledKernel(Fixed(0.0), EQ())
        @test kk + k == k + kk == k == sk + k == k + sk
        @test isa(k * kk, ZeroKernel)
        @test isa(kk * k, ZeroKernel)
        @test isa(k * sk, ZeroKernel)
        @test isa(sk * k, ZeroKernel)
        @test GPForecasting.is_not_noisy(kk)
        kkk = EQ()
        @test kkk + kk == kk + kkk == kkk
        @test isa(kkk * kk, ZeroKernel)
        @test isa(kk * kkk, ZeroKernel)
        @test isa(kkk * sk, ZeroKernel)
        @test isa(sk * kkk, ZeroKernel)
        @test isa(kk + kk, ZeroKernel)
        @test isa(kk * kk, ZeroKernel)
        @test isa(5 * kk, ZeroKernel)
        @test isa(zero(EQ()), ZeroKernel)
        @test isa(zero(Kernel), ZeroKernel)
    end

    @testset "Parameter" begin
        k = EQ() ▷ 0.7
        @test GPForecasting.is_not_noisy(k)
        @test k[:][1] ≈ -0.35667637251118145 atol = _ATOL_ rtol = _RTOL_
        k = EQ() ▷ Named(0.7, "named")
        @test k[:][1] ≈ -0.35667637251118145 atol = _ATOL_ rtol = _RTOL_
        k = EQ() ▷ Positive(0.7)
        @test k[:][1] ≈ -0.35667637251118145 atol = _ATOL_ rtol = _RTOL_
        k = EQ() ▷ Bounded(0.7, 0.1, 1.0)
        @test k[:][1] ≈ 0.6931455138937418 atol = _ATOL_ rtol = _RTOL_
        k = EQ() ▷ [1.0, 2.0, 3.0]
        @test k[:] ≈ [-1.0e-6, 0.6931466805598203, 1.0986119553347207] atol = _ATOL_ rtol = _RTOL_
        @test k([1. 2. 3.], [4. 5. 6.])[1] ≈ 0.002187491118182885 atol = _ATOL_ rtol = _RTOL_
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
            @test GPForecasting.is_not_noisy(k)
            @test k([1.,2.]) ≈ k([11.,12.]) atol = _ATOL_ rtol = _RTOL_
            k = periodicise(k ▷ [2.0, 3.0], [2π, 3π])
            @test k[:][1:2] ≈ [1.8378769072543897, 2.2433420684142087] atol = _ATOL_ rtol = _RTOL_
            @test k[:][3:4] ≈ [0.6931466805598203, 1.0986119553347207] atol = _ATOL_ rtol = _RTOL_
            @test isa(sprint(show, k), String)
        end

        x = collect(1.:0.1:5)
        @test periodicise(EQ(), 2.3)(x) ≈ exp.(-2sin.(π .* (x .- x') ./ 2.3).^2) # see
        # Rasmussen pg 92
    end

    @testset "Specified Quantity" begin
        # Using DataFrames/
        k = EQ()
        df = DataFrame(
            input1=[1.0, 2.0, 3.0],
            input2=[1.0, 1.0, 1.0],
            input3=[[1.0, 1.0], [2.0 , 1.0], [3.0, 1.0]],
        )
        sqk1 = k ← :input1
        sqk2 = k ← :input2
        sqk12 = (k ← :input1) * (k ← :input2)
        sqk3 = GPForecasting.takes_in(k, Fixed(:input3))
        @test GPForecasting.is_not_noisy(sqk1)
        @test !isMulti(sqk1)
        @test !(sqk1(df) ≈ sqk2(df))
        @test sqk2(df) ≈ ones(3, 3) atol = _ATOL_ rtol = _RTOL_
        @test isa(sprint(show, k), String)
        @test isa(var(sqk1, df), Vector)
        @test sqk12(df) ≈ sqk3(df) atol = _ATOL_ rtol = _RTOL_
        @test sqk12(df[1, :]) ≈ sqk3(DataFrame(df[1, :])) atol = _ATOL_ rtol = _RTOL_
        @test sqk3(df[1, :], df) ≈ sqk3(df, df[1, :])' atol = _ATOL_ rtol = _RTOL_

        # Using matrices.
        k1 = EQ()
        k2 = EQ() ▷ [1.0, 1.0]
        x = rand(4, 3)
        sqk1 = k1 ← 1
        sqk2 = k2 ← 1:2:3
        @test k1(x[:, 1]) ≈ sqk1(x) atol = _ATOL_ rtol = _RTOL_
        @test k2(x[:, [1, 3]]) ≈ sqk2(x) atol = _ATOL_ rtol = _RTOL_
        @test elwise(k1, x[:, 1]) ≈ elwise(sqk1, x) atol = _ATOL_ rtol = _RTOL_
    end

    @testset "Sum and Products" begin
        k_sum = EQ() + EQ()
        @test !isMulti(k_sum)
        @test GPForecasting.is_not_noisy(k_sum)
        k_prod = (2 * EQ()) * (3 * EQ())
        @test GPForecasting.is_not_noisy(k_prod)
        @test !isMulti(k_prod)
        @test k_sum([1.])[1, 1] ≈ 2.0 atol = _ATOL_ rtol = _RTOL_
        @test k_sum([1.], [2.])[1, 1] ≈ 1.2130613194252668 atol = _ATOL_ rtol = _RTOL_
        @test k_prod([1.], [1.])[1, 1] ≈ 6.0 atol = _ATOL_ rtol = _RTOL_
        @test k_prod([1.], [2.])[1, 1] ≈ 6 * 0.36787944117144233 atol = _ATOL_ rtol = _RTOL_
        @test isa(sprint(show, k_sum), String)
        @test isa(sprint(show, k_prod), String)
    end

    @testset "Set and Get" begin
        k = 5. * EQ() ▷ Fixed(2.) + Named(Fixed(2.), "variance") * EQ() ▷ Named(3., "scale")

        @test k[:] ≈ [1.6094377124340804, 1.0986119553347207] atol = _ATOL_ rtol = _RTOL_
        @test set(k, k[:] .+ log(2.))[:] ≈ [2.302584892994026, 1.791759135894666] atol = _ATOL_ rtol = _RTOL_
        @test k["variance"] ≈ 2.0 atol = _ATOL_ rtol = _RTOL_
        @test set(k, "variance" => 5.)["variance"] ≈ 5.0 atol = _ATOL_ rtol = _RTOL_
    end

    @testset "PosteriorKernel" begin
        pk = PosteriorKernel(ConstantKernel(), [1,2,3], Eye(3))
        @test !isMulti(pk)
        @test GPForecasting.is_not_noisy(pk)
        @test pk([1,2], [1,2]) ≈ [-2. -2.; -2. -2.] atol = _ATOL_ rtol = _RTOL_
        @test isa(sprint(show, pk), String)
    end

    @testset "Noise and Multi-output kernels" begin
        k = EQ() ← :in
        k2 = 5* EQ() ← :in
        mk = MultiKernel([k k2; k2 k])
        @test GPForecasting.is_not_noisy(mk)
        input = DataFrame([rand(5), rand(5)], [:in, :bs])
        @test isa(hourly_cov(mk, input), BlockDiagonal)
        @test isa(var(k, input), Vector)

        m = MultiKernel([EQ() 2*EQ(); EQ()+2 EQ()▷3])

        x = [1., 2., 3., 11., 22.]
        y = [1., 2., 3., 4., 5.]
        @test isMulti(m)
        @test m(x, 1, 1) ≈ EQ()(x) atol = _ATOL_ rtol = _RTOL_
        @test (EQ() + m)(x) ≈ MultiKernel(
            [EQ()+EQ() 2*EQ()+EQ(); EQ()+2+EQ() (EQ()▷3)+EQ()]
        )(x) atol = _ATOL_ rtol = _RTOL_
        @test (EQ() + m)(x) ≈ (1 * EQ() + m)(x) atol = _ATOL_ rtol = _RTOL_
        @test ((m + m) + m)(x) ≈ 3 .* m(x) atol = _ATOL_ rtol = _RTOL_
        @test diag(hourly_cov(m, x)) ≈ diag(m(x)) atol = _ATOL_ rtol = _RTOL_
        @test hourly_cov(m, x)[1:2, 1:2] ≈ m(x)[1:2, 1:2] atol = _ATOL_ rtol = _RTOL_
        @test !isapprox(hourly_cov(m, x)[2:4, 1:2], m(x)[2:4, 1:2], atol = _ATOL_, rtol = _RTOL_)
        @test isa(sprint(show, m), String)

        ox = Observed(x)
        oy = Observed(y)
        py = Latent(y)
        px = Latent(x)
        @test size(ox) == size(x)
        @test ox[1] isa Observed
        @test ox.val ≈ x
        mx = [Latent([1,2,3]), Observed([11, 22]), Latent([15])]

        nk = GPForecasting.NoiseKernel(EQ(), 12*DiagonalKernel()) # Giant noise to make it easy to spot
        @test !GPForecasting.is_not_noisy(nk)
        @test !isMulti(nk)
        @test nk(px, oy) ≈ nk(ox, py) ≈ nk(px, py)
        @test nk(ox, oy) ≈ nk(px, py) + [12 0 0 0 0; 0 12 0 0 0; 0 0 12 0 0; 0 0 0 0 0; 0 0 0 0 0] atol = _ATOL_ rtol = _RTOL_
        @test size(nk(ox, oy)) == (5, 5)
        @test diag(nk(mx)) ≈ [1.0, 1.0, 1.0, 13.0, 13.0, 1.0] atol = _ATOL_ rtol = _RTOL_
        @test nk(x, y)[1:2, 1:2] ≈ [13.0 1.0; 1.0 1.0] atol = _ATOL_ rtol = _RTOL_
        @test diag(hourly_cov(nk, ox)) ≈ diag(nk(ox)) atol = _ATOL_ rtol = _RTOL_
        @test diag(hourly_cov(nk, px)) ≈ diag(nk(px)) atol = _ATOL_ rtol = _RTOL_
        @test diag(hourly_cov(nk, x)) ≈ diag(nk(x)) atol = _ATOL_ rtol = _RTOL_
        @test hourly_cov(nk, x)[1:2, 1:2] ≈ nk(x)[1:2, 1:2] atol = _ATOL_ rtol = _RTOL_
        @test !isapprox(hourly_cov(nk, x)[2:4, 1:2], nk(x)[2:4, 1:2], atol = _ATOL_, rtol = _RTOL_)
        @test diag(hourly_cov(nk, mx)) ≈ diag(nk(mx)) atol = _ATOL_ rtol = _RTOL_
        @test isa(sprint(show, nk), String)

        nk = GPForecasting.NoiseKernel(m, 12*DiagonalKernel())
        @test isMulti(nk)
        @test nk(px, oy) ≈ nk(ox, py) ≈ nk(px, py)
        @test nk(ox, oy)[1:4, 1:2] ≈ (
            nk(px, py)[1:4, 1:2] .+ float.([12 12; 12 12; 0 0; 0 0])
        ) atol = _ATOL_ rtol = _RTOL_
        @test size(nk(ox, oy)) == (10, 10)
        @test diag(hourly_cov(nk, ox)) ≈ diag(nk(ox)) atol = _ATOL_ rtol = _RTOL_
        @test hourly_cov(nk, ox)[1:2, 1:2] ≈ nk(ox)[1:2, 1:2] atol = _ATOL_ rtol = _RTOL_
        @test !isapprox(hourly_cov(nk, ox)[2:4, 1:2], nk(ox)[2:4, 1:2], atol = _ATOL_, rtol = _RTOL_)
        @test diag(hourly_cov(nk, px)) ≈ diag(nk(px)) atol = _ATOL_ rtol = _RTOL_
        @test hourly_cov(nk, px)[1:2, 1:2] ≈ nk(px)[1:2, 1:2] atol = _ATOL_ rtol = _RTOL_
        @test !isapprox(hourly_cov(nk, px)[2:4, 1:2], nk(px)[2:4, 1:2], atol = _ATOL_, rtol = _RTOL_)
        @test diag(hourly_cov(nk, x)) ≈ diag(nk(x)) atol = _ATOL_ rtol = _RTOL_
        @test hourly_cov(nk, x)[1:4, 1:4] ≈ nk(x)[1:4, 1:4] atol = _ATOL_ rtol = _RTOL_
        @test !isapprox(hourly_cov(nk, x)[5:8, 1:4], nk(x)[5:8, 1:4], atol = _ATOL_, rtol = _RTOL_)
        @test diag(hourly_cov(nk, mx)) ≈ diag(nk(mx)) atol = _ATOL_ rtol = _RTOL_

        nk = GPForecasting.NoiseKernel(m, MultiKernel([12*DiagonalKernel() 0; 0 35*DiagonalKernel()]))
        @test isMulti(nk)
        @test nk(px, oy) ≈ nk(ox, py) ≈ nk(px, py)
        @test nk(ox, oy)[3:6, 3:4] ≈ (
            nk(px, py)[3:6, 3:4] .+ float.([12 0; 0 35; 0 0; 0 0])
        ) atol = _ATOL_ rtol = _RTOL_
        @test nk(ox, oy)[5:6, 5:6] ≈ (nk(px, py)[5:6, 5:6] .+ float.([12 0; 0 35])) atol = _ATOL_ rtol = _RTOL_
        @test nk(ox, oy)[6:10, 1:5] ≈ nk(px, py)[6:10, 1:5] atol = _ATOL_ rtol = _RTOL_
        @test nk(ox, oy)[1:5, 6:10] ≈ nk(px, py)[1:5, 6:10] atol = _ATOL_ rtol = _RTOL_
        @test size(nk(ox, oy)) == (10, 10)
        @test diag(hourly_cov(nk, ox)) ≈ diag(nk(ox)) atol = _ATOL_ rtol = _RTOL_
        @test hourly_cov(nk, ox)[1:2, 1:2] ≈ nk(ox)[1:2, 1:2] atol = _ATOL_ rtol = _RTOL_
        @test !isapprox(hourly_cov(nk, ox)[2:4, 1:2], nk(ox)[2:4, 1:2], atol = _ATOL_, rtol = _RTOL_)
        @test diag(hourly_cov(nk, px)) ≈ diag(nk(px)) atol = _ATOL_ rtol = _RTOL_
        @test hourly_cov(nk, px)[1:2, 1:2] ≈ nk(px)[1:2, 1:2] atol = _ATOL_ rtol = _RTOL_
        @test !isapprox(hourly_cov(nk, px)[2:4, 1:2], nk(px)[2:4, 1:2], atol = _ATOL_, rtol = _RTOL_)
        @test diag(hourly_cov(nk, x)) ≈ diag(nk(x)) atol = _ATOL_ rtol = _RTOL_
        @test hourly_cov(nk, x)[1:4, 1:4] ≈ nk(x)[1:4, 1:4] atol = _ATOL_ rtol = _RTOL_
        @test !isapprox(hourly_cov(nk, x)[5:8, 1:4], nk(x)[5:8, 1:4], atol = _ATOL_, rtol = _RTOL_)
        @test diag(hourly_cov(nk, mx)) ≈ diag(nk(mx)) atol = _ATOL_ rtol = _RTOL_

        k = NoiseKernel(EQ() ← :input1, 5.0 * DiagonalKernel())
        df = DataFrame([[1.,2.,3.], [1.,1.,1.]], [:input1, :input2])
        @test var(k, Latent(df)) == [1.0 1.0 1.0]'
        @test var(k, Observed(df)) == [6.0 6.0 6.0]'
        @test var(k, df) == [6.0, 1.0, 6.0, 1.0, 6.0, 1.0]

        k = LMMKernel(1, 3, 1e-2, rand(3, 1), EQ())
        @test var(k, df[:, :input1]) ≈ reshape(diag(k(df[:, :input1])), 3, 3)' atol = _ATOL_ rtol = _RTOL_
        kp = LMMPosKernel(k, df[:, :input1], rand(3, 3))
        @test var(kp, df[:, :input1]) ≈ reshape(diag(kp(df[:, :input1])), 3, 3)' atol = _ATOL_ rtol = _RTOL_
        k = LMMKernel(1, 3, 1e-2, rand(3, 1), EQ() ← :input1)
        @test var(k, df) ≈ reshape(diag(k(df)), 3, 3)' atol = _ATOL_ rtol = _RTOL_
        kp = LMMPosKernel(k, df, rand(3, 3))
        @test var(kp, df) ≈ reshape(diag(kp(df)), 3, 3)' atol = _ATOL_ rtol = _RTOL_
    end

    @testset "Matrix algebra" begin
        H = [3 2; 4 1; 2 5]
        @test isa(H * EQ(), MultiKernel)
        @test (H * EQ())(collect(1.:3.)) ≈ (EQ() * H)(collect(1.:3.)) atol = _ATOL_ rtol = _RTOL_
        @test H * ZeroKernel() == ZeroKernel() * H == Matrix{Kernel}([0 0; 0 0; 0 0])
        mk = MultiKernel([EQ() 3 * EQ(); 0 5 * EQ()])
        @test isa(H * (3 * mk), ScaledKernel)
        @test isMulti(H * (3 * mk))
        @test_throws DimensionMismatch (3 * mk) * H
        @test isMulti(H * (EQ() + RQ(5.5)))
        @test isMulti((EQ() + RQ(5.5)) * H)
        k = periodicise(EQ(), 2.)
        @test (H * k)([1., 2.]) ≈ (k * H)([3., 4.]) atol = _ATOL_ rtol = _RTOL_
    end

    @testset "Naive LMM" begin
        H = [1 2 3; 4 3 2; 4 4 4; 1 3 2; 7 6 3]
        nlmm = NaiveLMMKernel(3, 1e-2, H, EQ())
        # vnlmm = verynaiveLMMKernel(3, 5, 1e-2, H, EQ())
        lmm = LMMKernel(3, 5, 1e-2, H, [EQ(), EQ(), EQ()])
        @test isa(nlmm, NaiveLMMKernel)
        # @test isa(vnlmm, MultiKernel)
        x = collect(1:5)
        # @test nlmm(x) ≈ vnlmm(x) atol = _ATOL_ rtol = _RTOL_
        @test lmm(x) ≈ nlmm(x) atol = _ATOL_ rtol = _RTOL_
        @test isMulti(nlmm)
    end

    @testset "OLMM checks" begin
        A = ones(5,5) + 2Eye(5)
        U, S, V = svd(A)
        H = U * Diagonal(S)[:, 1:3]
        @test_throws ArgumentError OLMMKernel(
                3, # m
                5, # p
                1e-2, # σ²
                1e-2, # D
                Matrix(U[:, 1:3]), # H
                Matrix((U[:, 1:3])'), # P
                U[:, 1:3], # U
                Eye(3), # S_sqrt
                [EQ() for i in 1:3] # ks
        )
        @test_throws ArgumentError OLMMKernel(
                3, # m
                5, # p
                1e-2, # σ²
                1e-2, # D
                Matrix(U[:, 1:3]), # H
                Matrix((U[:, 1:3])'), # P
                U[:, 1:3], # U
                ones(3), # S_sqrt
                [EQ() for i in 1:2] # ks
        )
        @test_throws ArgumentError OLMMKernel(
                3, # m
                5, # p
                1e-2, # σ²
                1e-2, # D
                Matrix(U[:, 1:4]), # H
                Matrix((U[:, 1:3])'), # P
                U[:, 1:3], # U
                ones(3), # S_sqrt
                [EQ() for i in 1:3] # ks
        )
        @test_throws ArgumentError OLMMKernel(
                3, # m
                5, # p
                1e-2, # σ²
                1e-2, # D
                Matrix(U[:, 1:3]), # H
                Matrix((U[:, 1:4])'), # P
                U[:, 1:3], # U
                ones(3), # S_sqrt
                [EQ() for i in 1:3] # ks
        )
        @test_throws ArgumentError OLMMKernel(
                3, # m
                5, # p
                1e-2, # σ²
                1e-2, # D
                Matrix(U[:, 1:3]), # H
                Matrix((U[:, 1:3])'), # P
                U[:, 1:4], # U
                ones(3), # S_sqrt
                [EQ() for i in 1:3] # ks
        )
        @test_throws ArgumentError OLMMKernel(
                3, # m
                5, # p
                1e-2, # σ²
                1e-2, # D
                Matrix(U[:, 1:3]), # H
                Matrix((U[:, 1:3])'), # P
                U[:, 1:3], # U
                ones(4), # S_sqrt
                [EQ() for i in 1:3] # ks
        )
        @test_throws ArgumentError OLMMKernel(
                3, # m
                5, # p
                1e-2, # σ²
                1e-2, # D
                Matrix(U[:, 1:3]), # H
                Matrix((U[:, 1:3])'), # P
                U[:, 1:3], # U
                collect(1:3), # S_sqrt
                EQ() # ks
        )
        @test_throws ArgumentError OLMMKernel(
                3, # m
                5, # p
                1e-2, # σ²
                1e-2, # D
                Matrix(U[:, 1:3]), # H
                Matrix((U[:, 1:3])'), # P
                U[:, 1:3], # U
                collect(1:3), # S_sqrt
                [EQ() for i in 1:3] # ks
        )
    end

    @testset "LSOLMM" begin

        @testset "Constructor checks" begin-
            @test_throws ArgumentError LSOLMMKernel(
                    3, # m
                    4, # p
                    1e-2, # σ²
                    1e-2, # D
                    EQ(), # Hk
                    randn(3), # lat_pos
                    randn(5), # out_pos
                    [EQ() for i in 1:3], # ks
            )
            @test_throws ArgumentError LSOLMMKernel(
                    3, # m
                    0, # p
                    1e-2, # σ²
                    1e-2, # D
                    EQ(), # Hk
                    randn(4), # lat_pos
                    randn(5), # out_pos
                    [EQ() for i in 1:3], # ks
            )
            @test_throws ArgumentError LSOLMMKernel(
                3, # m
                5, # p
                1e-2, # σ²
                1e-2, # D
                EQ(), # Hk
                randn(4), # lat_pos
                randn(5), # out_pos
                [EQ() for i in 1:4], # ks
            )
        end

        @testset "1D latent positions" begin
            lat_pos = rand(3)
            out_pos = rand(5)

            A = rand(5, 3);
            U, S, V = svd(A);
            H = U * Diagonal(S);

            k = LSOLMMKernel(
                Fixed(3), Fixed(5), Fixed(0.02), Fixed([0.05 for i in 1:3]),
                stretch(EQ(), Positive(5.0)), lat_pos, out_pos,
                [stretch(EQ(), Positive(5.0)) for i in 1:3], Fixed(S)
            )

            ok = OLMMKernel(
                3, 5, 0.02, [0.05 for i in 1:3], H,
                [stretch(EQ(), Positive(5.0)) for i in 1:3]
            )
            k2 = LSOLMMKernel(stretch(EQ(), Positive(5.0)), lat_pos, out_pos, ok);
            @test unwrap(k.H) ≈ unwrap(k2.H)
            @test unwrap(k.P) ≈ unwrap(k2.P)
            @test unwrap(k.U) ≈ unwrap(k2.U)

            gp = GP(k)
            ngp = learn(gp, rand(4), rand(4, 5); trace=false)
            H = unwrap(ngp.k.H)
            P = unwrap(ngp.k.P)
            U = unwrap(ngp.k.U)
            S_sqrt = unwrap(ngp.k.S_sqrt)

            # Check that variables were duly updated
            @test !(unwrap(gp.k.Hk.stretch) ≈ unwrap(ngp.k.Hk.stretch))
            @test !(unwrap(gp.k.out_pos) ≈ unwrap(ngp.k.out_pos))
            @test !(unwrap(gp.k.lat_pos) ≈ unwrap(ngp.k.lat_pos))
            @test !(unwrap(gp.k.H) ≈ H)
            @test !(unwrap(gp.k.P) ≈ P)
            @test !(unwrap(gp.k.U) ≈ U)
            @test (unwrap(gp.k.S_sqrt) ≈ S_sqrt) # S_sqrt was Fixed

            # Check that the final mixing matrix has the right properties
            @test H ≈ U * Diagonal(S_sqrt)
            @test diag(H' * H) ≈ S_sqrt.^2
            @test diag(P * H) ≈ ones(size(P, 1))

            # Let S_sqrt change
            k.olmm.S_sqrt = Positive(S)
            ngp = learn(GP(k), rand(4), rand(4, 5); trace=false)
            S_sqrt = unwrap(ngp.k.S_sqrt)
            @test !(unwrap(gp.k.S_sqrt) ≈ S_sqrt)
        end

        @testset "3D latent positions" begin
            k = LSOLMMKernel(
                Fixed(3), Fixed(5), Fixed(0.02), Fixed([0.05 for i in 1:3]),
                stretch(EQ(), Positive([5.0, 3.0, 4.0])),
                rand(3, 3), rand(5, 3),
                [stretch(EQ(), Positive(5.0)) for i in 1:3]
            )
            gp = GP(k)

            ngp = learn(gp, rand(4), rand(4, 5); trace=false)
            H = unwrap(ngp.k.H)
            P = unwrap(ngp.k.P)
            U = unwrap(ngp.k.U)
            S_sqrt = unwrap(ngp.k.S_sqrt)

            # Check that variables were duly updated
            @test !(unwrap(gp.k.Hk.stretch) ≈ unwrap(ngp.k.Hk.stretch))
            @test !(unwrap(gp.k.out_pos) ≈ unwrap(ngp.k.out_pos))
            @test !(unwrap(gp.k.lat_pos) ≈ unwrap(ngp.k.lat_pos))
            @test !(unwrap(gp.k.H) ≈ H)
            @test !(unwrap(gp.k.P) ≈ P)
            @test !(unwrap(gp.k.U) ≈ U)
            @test (unwrap(gp.k.S_sqrt) ≈ S_sqrt) # S_sqrt is Fixed by default

            # Check that the final mixing matrix has the right properties
            @test H ≈ U * Diagonal(S_sqrt)
            @test diag(H' * H) ≈ S_sqrt.^2
            @test diag(P * H) ≈ ones(size(P, 1))

            pgp = condition(ngp, rand(4), rand(4, 5))
            @test pgp isa GP
            @test pgp.k isa LSOLMMKernel
            gaus = pgp(rand(4))
            @test gaus isa Gaussian
        end
    end

    @testset "GOLMMKernel sampling, learning and inference" begin

        ### sample some data from a GOLMM
        p, m = 4, 2
        ks = [stretch(EQ(), 3.0), stretch(EQ(), 7.0)]
        group_embs = [1.0, 1.01, 1.1, 5.0]
        group_k = stretch(EQ(), 2.0)

        k_golmm = GOLMMKernel(m, p, 1e-6, 1e-6, ks, group_k, group_embs)
        gp = GP(k_golmm)

        tmin, tmax, step = 0.0, 20.0, 0.05
        ts = collect(tmin:step:tmax)
        ys = sample(gp(ts))

        @test size(ys) == (Int((tmax - tmin) / step + 1), p)
        #@test all(abs.(ys[:, 1] - ys[:, 3]) .< 1e-2)
        #@test all(abs.(ys[:, 1] - ys[:, 3]) .< abs.(ys[:, 1] - ys[:, 4]))

        ### create a new GOLMM and fit it to the data
        @testset "Learning" begin
            group_embs_init = [1., 2., 3., 4.]
            eq_ls_init = 1.0
            k_golmm_init = GOLMMKernel(
                m,
                p,
                Positive(0.01),
                Positive(0.001),
                ks,
                stretch(EQ(), eq_ls_init),
                group_embs_init,
            )

            gp = GP(k_golmm_init)
            gp = learn(gp, ts, ys, mle_obj, its=50, trace=false)

            # test that group embeddings have been updated
            @test !all(
                isapprox.(
                    group_embs_init,
                    gp.k.group_embeddings;
                    atol = _ATOL_,
                    rtol = _RTOL_,
                )
            )

            # test that group kernel has been updated
            @test !isapprox(
                eq_ls_init,
                GPForecasting.unwrap(gp.k.group_kernel.stretch);
                atol = _ATOL_,
                rtol = _RTOL_,
            )
        end

        ### try to do inference, make sure it does not break
        @testset "Inference" begin
            posterior_gp = condition(gp, ts, ys);
            @test isa(posterior_gp, GP)
        end
    end

    @testset "ManifoldKernel" begin
        k = EQ()

        activation = sigmoid
        l1 = NNLayer(randn(6, 1), 1e-2 .* randn(6), Fixed(activation))
        l2 = NNLayer(randn(2, 6), 1e-2 .* randn(2), Fixed(activation))
        nn = GPFNN([l1, l2])
        mk = ManifoldKernel(k, nn)

        x = rand(50)
        @test GPForecasting.is_not_noisy(mk)
        @test isa(mk(x), AbstractMatrix)
        @test mk(x) ≈ mk(x, x) atol = _ATOL_ rtol = _RTOL_
        @test diag(mk(x)) ≈ var(mk, x) atol = _ATOL_ rtol = _RTOL_
        @test hourly_cov(mk, x) ≈ Diagonal(var(mk, x)) atol = _ATOL_ rtol = _RTOL_

        k = NoiseKernel(1.0 * stretch(EQ(), Positive([1.0, 1.0])), Fixed(1e-2) * DiagonalKernel())
        x = Observed(x)
        mk = ManifoldKernel(k, nn)
        @test !GPForecasting.is_not_noisy(mk)
        @test isa(mk(x), AbstractMatrix)
        @test mk(x) ≈ mk(x, x) atol = _ATOL_ rtol = _RTOL_
        @test diag(mk(x)) ≈ var(mk, x) atol = _ATOL_ rtol = _RTOL_
        @test hourly_cov(mk, x) ≈ Diagonal(var(mk, x)) atol = _ATOL_ rtol = _RTOL_
    end

    @testset "hourly_cov" begin
        n = 100;
        p = 5;
        m = 3;
        d = 3;
        x = rand(n, d);
        y = rand(n, p);
        H = rand(p, m);
        U, S, V = svd(H);
        H = U * Diagonal(sqrt.(S))
        k = [(EQ() ▷ 10.0) for i=1:m];
        gp = GP(OLMMKernel(m, p, 0.1, 0.0, H, k));
        olmm = condition(gp, x, y);
        @test size(hourly_cov(olmm.k, x[1:10, :])) == (50, 50)

        k = [(EQ() ▷ [10.0, 10.0, 10.0]) for i=1:m];
        gp = GP(OLMMKernel(m, p, 0.1, 0.0, H, k));
        olmm = condition(gp, x, y);
        size(olmm.m(x[1:10, :]))
        size(olmm.k(x[1:10, :]))
        @test size(hourly_cov(olmm.k, x[1:10, :])) == (50, 50)
    end
end
