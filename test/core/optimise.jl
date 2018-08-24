@testset "Optimise" begin
    @test sum(minimise(
        x -> x[1]^2+abs(x[2]),
        [100., 900.],
        trace=true,
        alphaguess=LineSearches.InitialQuadratic(),
        linesearch=LineSearches.HagerZhang(),
    )) < 1e-5

    gp = GP(0, EQ() ▷ 10)
    x = collect(1.0:10.0);
    y = 2 .* x + 1e-1*randn();
    ngp = learn(gp, x, y, objective, trace=true)
    @test 1.5 < ngp.k.stretch.p < 3.5

    # test the OLMM learn method
    xs = hcat([sin.(2π*collect(0:0.1:2)./i) for i in 1:3]...)
    A = ones(5,5) + 2eye(5)
    U, S, V = svd(A)
    H = U * diagm(S)[:, 1:3]
    y = (H * xs')'
    gp = GP(OLMMKernel(3, 5, 10., 1e-2, H, [periodicise(EQ(), i) for i in 1:3]))
    x = collect(0:0.1:2)
    ngp = learn(gp, x, y, objective, opt_U=true, its=3, trace=false)
    Ug = GPForecasting.unwrap(gp.k.U)
    @test sum(Ug' * Ug) ≈ 3.0 atol = _ATOL_
    ngp = learn(gp, x, y, objective, its=3, K_U_cycles=2, trace=false)
    Ug = GPForecasting.unwrap(gp.k.U)
    @test sum(Ug' * Ug) ≈ 3.0 atol = _ATOL_

    # Test that the summary is outputted when called
    out = learn_summary(gp, x, y, objective, its=3, trace=false)
    @test size(out, 1) == 2
    @test isa(out[2], GP) == true
end
