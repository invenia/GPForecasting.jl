# Here we want to ensure that stuff is differentiable
@testset "Nabla" begin
    # Check gradients of `pairwise_dist`.
    for (is_x, is_y) in (((4,), (6,)), ((4, 2), (6, 2)))
        rx, dx = randn(is_x...), randn(is_x...)
        ry, dy = randn(is_y...), randn(is_y...)
        @test check_errs(x -> pairwise_dist(x, ry), randn(is_x[1], is_y[1]), rx, dx)
        @test check_errs(y -> pairwise_dist(rx, y), randn(is_x[1], is_y[1]), ry, dy)
    end

    # Check gradients of `elwise_dist`.
    for (is_x, is_y) in (((4,), (4,)), ((4, 2), (4, 2)))
        rx, dx = randn(is_x...), randn(is_x...)
        ry, dy = randn(is_y...), randn(is_y...)
        @test check_errs(x -> elwise_dist(x, ry), randn(is_x[1]), rx, dx)
        @test check_errs(y -> elwise_dist(rx, y), randn(is_x[1]), ry, dy)
    end

    # Check gradients of `sq_elwise_dist`.
    for (is_x, is_y) in (((4,), (4,)), ((4, 2), (4, 2)))
        rx, dx = randn(is_x...), randn(is_x...)
        ry, dy = randn(is_y...), randn(is_y...)
        @test check_errs(x -> sq_elwise_dist(x, ry), randn(is_x[1]), rx, dx)
        @test check_errs(y -> sq_elwise_dist(rx, y), randn(is_x[1]), ry, dy)
    end

    gp = GP(0, EQ() ▷ 10)
    x = collect(1.0:10.0)
    y = 2 .* x .+ 1e-1*randn()
    obj = mle_obj(gp, x, y)

    @test !(
        ∇(obj)(GPForecasting.pack(Positive(2.5)))[1] ≈
        ∇(obj)(GPForecasting.pack(Positive(1e5)))[1]
    )

    μ = [12., 3., 9.]
    Σ = Eye(3)
    n = Gaussian(μ, Σ)
    obj2(x) = logpdf(n, x)
    @test !(∇(obj2)([1., 2., 3.])[1] ≈ ∇(obj2)([11., 2., 32.])[1])

    xs = hcat([sin.(collect(1:6)./i) for i in 1:3]...)
    H = [1 2 3; 4 3 2; 4 4 4; 1 3 2; 7 6 3]
    y = (H * xs')'
    gp = GP(LMMKernel(3, 5, 1e-2, H, EQ()))
    x = collect(1.0:6.0)
    obj3 = mle_obj(gp, x, y)
    @test isa(obj3(gp.k[:]), Float64)
    @test isa(∇(obj3)(gp.k[:])[1][1], Float64)
    @test !(∇(obj3)(gp.k[:])[1] ≈ ∇(obj3)(3 .* gp.k[:])[1])

    xs = hcat([sin.(2π*collect(0:0.1:2)./i) for i in 1:3]...)
    A = ones(5,5) + 2Eye(5)
    U, S, V = svd(A)
    H = U * Diagonal(S)[:, 1:3]
    y = (H * xs')'
    gp = GP(OLMMKernel(3, 5, 10., 1e-2, H, [periodicise(EQ(), i) for i in 1:3]))
    x = collect(0:0.1:2)
    obj3 = mle_obj(gp, x, y)
    @test isa(obj3(gp.k[:]), Float64)
    @test isa(∇(obj3)(gp.k[:])[1][1], Float64)
    @test !(∇(obj3)(gp.k[:])[1] ≈ ∇(obj3)(3 .* gp.k[:])[1])

    # Manifold GPs
    k = EQ()
    activation = sigmoid
    l1 = NNLayer(randn(6, 1), 1e-2 .* randn(6), Fixed(activation))
    l2 = NNLayer(randn(2, 6), 1e-2 .* randn(2), Fixed(activation))
    nn = GPFNN([l1, l2])
    mk = ManifoldKernel(k, nn)
    obj = mle_obj(GP(mk), rand(5), rand(5))
    @test isa(∇(obj)(mk[:])[1], Vector)
end
