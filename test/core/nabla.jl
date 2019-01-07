using FDM

# Here we want to ensure that stuff is differentiable
@testset "Nabla" begin
    # Check gradients of `pairwise_dist`.
    for (is_x, is_y) in (((4,), (6,)), ((4, 2), (6, 2)))
        x, y = randn(is_x...), randn(is_y...)
        v_x, v_y = randn(is_x...), randn(is_y...)
        z̄ = randn(is_x[1], is_y[1])
        grad_numerical_x = dot(z̄, FDM.central_fdm(3, 1)(ε -> pairwise_dist(x .+ ε .* v_x, y)))
        grad_numerical_y = dot(z̄, FDM.central_fdm(3, 1)(ε -> pairwise_dist(x, y .+ ε .* v_y)))
        grads = ∇((x, y) -> dot(pairwise_dist(x, y), z̄))(x, y)
        grad_∇_x, grad_∇_y = dot(v_x, grads[1]), dot(v_y, grads[2])
        @test grad_numerical_x ≈ grad_∇_x atol = _ATOL_
        @test grad_numerical_y ≈ grad_∇_y atol = _ATOL_
    end

    gp = GP(0, EQ() ▷ 10)
    x = collect(1.0:10.0)
    y = 2 .* x .+ 1e-1*randn()
    obj = objective(gp, x, y)

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
    obj3 = objective(gp, x, y)
    @test isa(obj3(gp.k[:]), Float64)
    @test isa(∇(obj3)(gp.k[:])[1][1], Float64)
    @test !(∇(obj3)(gp.k[:])[1] ≈ ∇(obj3)(3 .* gp.k[:])[1])

    xs = hcat([sin.(2π*collect(0:0.1:2)./i) for i in 1:3]...)
    A = ones(5,5) + 2Eye(5)
    U, S, V = svd(A)
    H = U * diagm(S)[:, 1:3]
    y = (H * xs')'
    gp = GP(OLMMKernel(3, 5, 10., 1e-2, H, [periodicise(EQ(), i) for i in 1:3]))
    x = collect(0:0.1:2)
    obj3 = objective(gp, x, y)
    @test isa(obj3(gp.k[:]), Float64)
    @test isa(∇(obj3)(gp.k[:])[1][1], Float64)
    @test !(∇(obj3)(gp.k[:])[1] ≈ ∇(obj3)(3 .* gp.k[:])[1])
end
