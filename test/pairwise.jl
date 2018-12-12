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
end
