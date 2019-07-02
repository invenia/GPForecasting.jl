@testset "Optimise" begin
   @testset "Optim.jl" begin
        @test sum(minimise(
            x -> x[1]^2+abs(x[2]),
            [100., 900.],
            trace=false,
            alphaguess=LineSearches.InitialQuadratic(),
            linesearch=LineSearches.HagerZhang(),
        )) < 1e-5

        @test minimise(
            x -> (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2,
            [10., -10.],
            trace=false,
            algorithm=Optim.ConjugateGradient,
            alphaguess=LineSearches.InitialQuadratic(),
            linesearch=LineSearches.HagerZhang(),
           ) ≈ [1.0, 1.0] atol = _ATOL_
   end

    @testset "Learn" begin
        gp = GP(0, EQ() ▷ 10)
        x = collect(1.0:10.0);
        y = 2 .* x .+ 1e-1 * randn()
        ngp = learn(gp, x, y, mle_obj, trace=false)
        @test 1.5 < ngp.k.stretch.p < 3.5

        # test the OLMM learn method
        xs = hcat([sin.(2π*collect(0:0.1:2)./i) for i in 1:3]...)
        A = ones(5,5) + 2Eye(5)
        U, S, V = svd(A)
        H = U * Diagonal(S)[:, 1:3]
        y = (H * xs')'
        gp = GP(OLMMKernel(3, 5, 10., 1e-2, H, [periodicise(EQ(), i) for i in 1:3]))
        gp.k.H = Fixed(gp.k.H)
        x = collect(0:0.1:2)
        ngp = learn(gp, x, y, mle_obj, opt_U=true, its=3, trace=false)
        Ug = GPForecasting.unwrap(gp.k.U)
        @test sum(Ug' * Ug) ≈ 3.0 atol = _ATOL_
        ngp = learn(gp, x, y, mle_obj, its=3, K_U_cycles=2, trace=false)
        Ug = GPForecasting.unwrap(gp.k.U)
        @test sum(Ug' * Ug) ≈ 3.0 atol = _ATOL_

        # Test that the summary is outputted when called
        out = learn_summary(gp, x, y, mle_obj, its=3, trace=false)
        @test size(out, 1) == 2
        @test isa(out[2], GP) == true

        # Unidimensional sGPs
        x = collect(0:0.01:2)
        y = sin.(4π * x) .+ 1e-1 .* randn(length(x))
        gp = GP(periodicise(EQ(), 1.0))
        Xm_i = [0.0, 1.0, 2.0]
        sgp, Xm, σ² = learn_sparse(gp, x, y, Fixed(Xm_i), Positive(0.01), its=20, trace=false)
        @test GPForecasting.unwrap(Xm) ≈ Xm_i atol=_ATOL_
        sgp, Xm, σ² = learn_sparse(gp, x, y, Xm_i, Positive(0.01), its = 20, trace=false)
        @test !isapprox(GPForecasting.unwrap(Xm), Xm_i, atol=_ATOL_)
        sgp, Xm, σ² = learn_sparse(gp, x, y, Xm_i, Fixed(0.01), its = 20, trace=false)
        @test GPForecasting.unwrap(σ²) == 0.01
        # Multidimensional sGPs
        y = [y y]
        gp = GP(OLMMKernel(2, 2, 0.01, 0.01, [1 0; 0 1], [periodicise(EQ(), 1.0), periodicise(EQ(), 1.0)]))
        Xm_i = [0.0, 1.0, 2.0]
        sgp, Xm, σ² = learn_sparse(gp, x, y, Fixed(Xm_i), Positive(0.01), its=20, trace=false)
        @test GPForecasting.unwrap(Xm) ≈ Xm_i atol=_ATOL_
        sgp, Xm, σ² = learn_sparse(gp, x, y, Xm_i, Positive(0.01), its = 20, trace=false)
        @test !isapprox(GPForecasting.unwrap(Xm), Xm_i, atol=_ATOL_)
        sgp, Xm, σ² = learn_sparse(gp, x, y, Xm_i, Fixed(0.01), its = 20, trace=false)
        @test GPForecasting.unwrap(σ²) == 0.01
        # Multidmensional inputs
        x = [x reverse(x)]
        y = sin.(4π * sum(x, dims=2)) .+ 1e-1 .* randn(size(x, 1)); y = dropdims(y, dims=2)
        gp = GP(periodicise(EQ(), 1.0))
        Xm_i = [0.0, 1.0, 2.0]
        Xm_i = [Xm_i reverse(Xm_i)]
        sgp, Xm, σ² = learn_sparse(gp, x, y, Fixed(Xm_i), Positive(0.01), its=20, trace=false)
        @test GPForecasting.unwrap(Xm) ≈ Xm_i atol=_ATOL_
        sgp, Xm, σ² = learn_sparse(gp, x, y, Xm_i, Positive(0.01), its = 20, trace=false)
        @test !isapprox(GPForecasting.unwrap(Xm), Xm_i, atol=_ATOL_)
        sgp, Xm, σ² = learn_sparse(gp, x, y, Xm_i, Fixed(0.01), its = 20, trace=false)
        @test GPForecasting.unwrap(σ²) == 0.01
    end
end
