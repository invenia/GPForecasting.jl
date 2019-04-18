@testset "Pairwise" begin
    z = [1.0 2.0; 3.0 4.0]

    @test pairwise_dist(z, z) ≈ [0.0 2.8284271247461903; 2.8284271247461903 0.0] atol = _ATOL_

    seed!(314159265)
    x = rand(10,3)
    y = rand(5,3)
    u = rand(10)
    v = rand(5)
    t = rand(1)[1]

    @test pairwise_dist(x, y).^2 ≈ sq_pairwise_dist(x, y)
    f(t) = dot(u, pairwise_dist(x./t, y./t)*v)
    @test ∇(f)(t)[1] ≈ central_fdm(2, 1)(f, t) atol = _ATOL_
    f(t) = dot(u, pairwise_dist(x[:,1]./t, y[:,1]./t)*v)
    @test ∇(f)(t)[1] ≈ central_fdm(2, 1)(f, t) atol = _ATOL_

    @testset "sq_pairwise_dist" begin
        rng = MersenneTwister(1)
        n = 10
        x = randn(rng, n)
        y = randn(rng, n)
        @test isapprox(
            sq_pairwise_dist(x, y),
            sq_pairwise_dist(reshape(x, n, 1), reshape(y, n, 1));
            atol=_ATOL_,
        )
    end
    @testset "Sensitivities" begin
        rng = MersenneTwister(1)
        n = 10
        for sz in [(n, n), (n,)]
            rx = randn(rng, sz)
            ry = randn(rng, sz)
            @test check_errs(x -> sq_pairwise_dist(x, ry), randn(rng, n, n), rx, ry)
            @test check_errs(y -> sq_pairwise_dist(rx, y), randn(rng, n, n), rx, ry)
        end
    end
end
