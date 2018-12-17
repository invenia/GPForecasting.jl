@testset "Synthetic" begin
    _TOL_ = 2.0e-3

    function mse(means, y_true)
        return mean((y_true .- means).^2)
    end

    function logpdf(means, vars, y_true)
        return -0.5 * mean(log.(2π .* vars)) .- 0.5 * mean((y_true .- means).^2 ./ vars)
    end

    function gpforecasting(m, k, x_train, y_train, x_test, y_test)
        gp = GP(m, k)
        b = condition(gp, Observed(x_train), y_train)
        pgp = learn(gp, Observed(x_train), y_train, objective, trace=false)
        a = condition(pgp, Observed(x_train), y_train)
        return [
            mse(b.m(Latent(x_test)), y_test),
            logpdf(b.m(Latent(x_test)), diag(b.k(Latent(x_test))), y_test),
            mse(a.m(Latent(x_test)), y_test),
            logpdf(a.m(Latent(x_test)), diag(a.k(Latent(x_test))), y_test),
        ]
    end

    @testset "One dimensional problems" begin
        seed!(314159265)
        n = 50
        f_1d(x) = sin.(x) .* sin.(2.0 * x) + 0.25 * rand(n)
        x_train = sort(2pi .* rand(n))
        x_test = sort(2pi .* rand(n))
        y_train = f_1d(x_train)
        y_test = f_1d(x_test)

        m = ZeroMean()

        @testset "Kernels" begin
            k = NoiseKernel(2.0 * (EQ() ▷ 10.0), 0.001 * DiagonalKernel())
            @test gpforecasting(m, k, x_train, y_train, x_test, y_test) ≈
                [0.14825538985245576, -1476.7191748828689,
                0.006339639909681591, 0.572904063004879] atol = _TOL_
            k = NoiseKernel(2.0 * periodicise(EQ() ▷ 10.0, 2π), 0.001 * DiagonalKernel())
            @test gpforecasting(m, k, x_train, y_train, x_test, y_test) ≈
                [0.13079765742457203, -760.2721834283418,
                0.0101082770781702, -0.36499061897453666] atol = _TOL_
            k = NoiseKernel(2.0 * (RQ(2.0) ▷ 5.0), 0.001 * DiagonalKernel())
            @test gpforecasting(m, k, x_train, y_train, x_test, y_test) ≈
                [0.12226859947646129, -771.9774842426579,
                0.006339647377187701, 0.5728382447611549] atol = _TOL_
            k = NoiseKernel(2.0 * (MA(5/2) ▷ 10.0), 0.001 * DiagonalKernel())
            @test gpforecasting(m, k, x_train, y_train, x_test, y_test) ≈
                [0.110754882488376, -625.1263034474591,
                0.0058380317446004625, 0.8934805590278987] atol = _TOL_
            k = NoiseKernel(2.0 * (EQ() ▷ 10.0 + periodicise(EQ() ▷ 10.0, 2π) * RQ(2.0) ▷ 10.0)
                + MA(5/2) ▷ 10.0, 0.001 * DiagonalKernel())
            @test gpforecasting(m, k, x_train, y_train, x_test, y_test) ≈
                [0.11769231376321873, -745.8368568753356,
                0.005939526953344254, 0.8604613342026686] atol = _TOL_
        end

        @testset "Parameters" begin
            k = NoiseKernel(2.0 * (EQ() ▷ Positive(10.0)), 0.001 * DiagonalKernel())
            @test gpforecasting(m, k, x_train, y_train, x_test, y_test) ≈
                [0.14825538985245576, -1476.7191748828689,
                0.006339639909681591, 0.572904063004879] atol = _TOL_
            k = NoiseKernel(2.0 * (EQ() ▷ Fixed(10.0)), 0.001 * DiagonalKernel())
            @test gpforecasting(m, k, x_train, y_train, x_test, y_test) ≈
                [0.14825538985245576, -1476.7191748828689,
                0.15858203241160362, -9.226749455378185] atol = _TOL_
            k = NoiseKernel(2.0 * (EQ() ▷ Bounded(10.0, 9.0, 11.0)), 0.001 * DiagonalKernel())
            @test gpforecasting(m, k, x_train, y_train, x_test, y_test) ≈
                [0.14825538985245576, -1476.7191748828689,
                0.15799166609934295, -9.145120578228152] atol = _TOL_
            k = NoiseKernel(Fixed(2.0) * (EQ() ▷ Positive(10.0) +
                periodicise(EQ() ▷ 10.0, Fixed(2π)) * RQ(Bounded(2.0, 1.5, 4.0)) ▷ 10.0) +
                MA(5/2) ▷ 10.0, 0.001 * DiagonalKernel())
            @test gpforecasting(m, k, x_train, y_train, x_test, y_test) ≈
                [0.11769231376321873, -745.8368568753356,
                0.0059346146440780664, 0.8602727132150634] atol = _TOL_
        end

        @testset "Sampled from GP" begin
            k = NoiseKernel(EQ() ▷ 0.7, 0.01 * DiagonalKernel())
            p = condition(GP(m, k), Observed(x_train), y_train)
            y_sample = vec(sample(p(Observed(x_train))))
            k = NoiseKernel(EQ() ▷ 2.0, 0.02 * DiagonalKernel())
            pgp = learn(GP(m, k), Observed(x_train), y_sample, objective, trace=false)
            @test exp.(pgp.k[:]) ≈ [0.787194, 0.00574296] atol = _TOL_
        end
    end

    @testset "Multidimensional input problems" begin
        seed!(314159265)
        n = 100
        d = 3
        s = collect(1.0:1.0:d)
        f_Md(x) = proddims(sin.(x .* s'), 2)[:] + 0.1 * rand(n)
        x_train = 2pi .* rand(n, d)
        x_test = 2pi .* rand(n, d)
        y_train = f_Md(x_train)
        y_test = f_Md(x_test)

        m = ZeroMean()

        k = NoiseKernel(2.0 * (EQ() ▷ [2.0 for i=1:d]), 0.001 * DiagonalKernel())
        @test gpforecasting(m, k, x_train, y_train, x_test, y_test) ≈
            [0.5375517536959576, -35.84754149283952,
            0.0981019604966694, -0.2253175147006799] atol = _TOL_
        k = NoiseKernel(2.0 * periodicise(EQ() ▷ [2.0 for i=1:d],
                [2π for i=1:d]), 0.001 * DiagonalKernel())
        @test gpforecasting(m, k, x_train, y_train, x_test, y_test) ≈
            [0.34025623421380025, -21.45435835111746,
            0.1309171532520091, -0.4750931597229358] atol = _TOL_
        k = NoiseKernel(2.0 * (RQ(Fixed(2.0)) ▷ [2.0 for i=1:d]), 0.001 * DiagonalKernel())
        @test gpforecasting(m, k, x_train, y_train, x_test, y_test) ≈
            [0.45069841363296914, -8.104167427835442,
            0.10054178097162188, -0.24119368608047453] atol = _TOL_
        k = NoiseKernel(2.0 * (MA(5/2) ▷ [2.0 for i=1:d]), 0.001 * DiagonalKernel())
        @test gpforecasting(m, k, x_train, y_train, x_test, y_test) ≈
            [0.2387315118745936, -0.875735261494879,
            0.09881765784593713, -0.23168536130876932] atol = _TOL_
    end

    @testset "Multidimensional output problems" begin
        seed!(314159265)

        n = 100
        p = 5

        C = rand(p, p)

        x = sort(rand(2n))
        y = zeros(2n, p)
        y[1,:] = rand(p)
        for i=2:2n
            y[i,:] = y[i-1,:] .+ rand(p) .- 0.5
        end

        y = y*C
        x_train = x[1:2:end]
        x_test = x[2:2:end]
        y_train = y[1:2:end,:]
        y_test = y[2:2:end,:]
        mean_y_train = meandims(y_train, 1)
        std_y_train = stddims(y_train, 1)
        y_train = (y_train .- mean_y_train) ./ std_y_train
        y_test = (y_test .- mean_y_train) ./ std_y_train

        H = Eye(p)
        k = [(EQ() ▷ 10.0) for i=1:5]
        gp = GP(LMMKernel(Fixed(5), Fixed(p), Positive(0.001), Fixed(H), k))
        b = condition(gp, x_train, y_train).m(x_test)
        gp = learn(gp, x_train, y_train, objective, its=50, trace=false)
        a = condition(gp, x_train, y_train).m(x_test)
        @test mse(vec(b), vec(y_test)) ≈ 0.927596879880526  atol = _TOL_
        @test mse(vec(a), vec(y_test)) ≈ 0.8087681056630357 atol = _TOL_

        U, S, V = svd(cov(y_train))
        H = U * Diagonal(sqrt.(S))
        k = [(EQ() ▷ 10.0) for i=1:5]
        gp = GP(LMMKernel(Fixed(5), Fixed(p), Positive(0.001), Fixed(H), k))
        b = condition(gp, x_train, y_train).m(x_test)
        gp = learn(gp, x_train, y_train, objective, its=50, trace=false)
        a = condition(gp, x_train, y_train).m(x_test)
        @test mse(vec(b), vec(y_test)) ≈ 0.8751756874359901  atol = _TOL_
        @test mse(vec(a), vec(y_test)) ≈ 0.10721174421720554 atol = _TOL_

        H = (U * Diagonal(sqrt.(S)))[:,1:3]
        k = [(EQ() ▷ 10.0) for i=1:3]
        gp = GP(LMMKernel(Fixed(3), Fixed(p), Positive(0.001), Fixed(H), k))
        b = condition(gp, x_train, y_train).m(x_test)
        gp = learn(gp, x_train, y_train, objective, its=50, trace=false)
        a = condition(gp, x_train, y_train).m(x_test)
        @test mse(vec(b), vec(y_test)) ≈ 0.8754142897078707  atol = _TOL_
        @test mse(vec(a), vec(y_test)) ≈ 0.10636472286495949 atol = _TOL_

        H = (U * Diagonal(sqrt.(S)))[:,1:1]
        k = [(EQ() ▷ 10.0)]
        gp = GP(LMMKernel(Fixed(1), Fixed(p), Positive(0.001), Fixed(H), k))
        b = condition(gp, x_train, y_train).m(x_test)
        gp = learn(gp, x_train, y_train, objective, its=50, trace=false)
        a = condition(gp, x_train, y_train).m(x_test)
        @test mse(vec(b), vec(y_test)) ≈ 0.8846475452351197  atol = _TOL_
        @test mse(vec(a), vec(y_test)) ≈ 0.11874106402148683 atol = _TOL_

        seed!(314159265)
        H = rand(p, 7)
        k = [(EQ() ▷ 10.0) for i=1:7]
        gp = GP(LMMKernel(Fixed(7), Fixed(p), Positive(0.001), Fixed(H), k))
        b = condition(gp, x_train, y_train).m(x_test)
        gp = learn(gp, x_train, y_train, objective, its=50, trace=false)
        a = condition(gp, x_train, y_train).m(x_test)
        @test mse(vec(b), vec(y_test)) ≈ 0.8202143612660611  atol = _TOL_
        @test mse(vec(a), vec(y_test)) ≈ 0.20612005452907958 atol = _TOL_
    end

    @testset "Comparison of LMM/OLMM/SOLMM models" begin
        seed!(314159265)

        n = 100
        p = 5
        m = 3

        obs_noise = 0.1
        lat_noise = 0.02

        C = rand(p, p)

        x = sort(rand(2n))
        y = zeros(2n, p)
        y[1,:] = rand(p)
        for i=2:2n
            y[i,:] = y[i-1,:] .+ rand(p) .- 0.5
        end

        y = y*C
        x_train = x[1:2:end]
        x_test = x[2:2:end]
        y_train = y[1:2:end,:]
        y_test = y[2:2:end,:]
        mean_y_train = meandims(y_train, 1)
        std_y_train = stddims(y_train, 1)

        y_train_standardised = (y_train .- mean_y_train) ./ std_y_train

        U, S, V = svd(cov(y_train_standardised))
        H = (U * Diagonal(sqrt.(S)))[:,1:m]

        k1 = [(EQ() ▷ 10.0) for i=1:m]
        k2 = [(EQ() ▷ (10.0 / i)) for i=1:m]

        # lat_noise = 0.0

        # LMM
        gp = GP(LMMKernel(Fixed(m), Fixed(p), Positive(obs_noise), Fixed(H), k1))
        lmm = condition(gp, x_train, y_train_standardised)
        m_lmm = lmm.m(x_test)
        k_lmm = lmm.k(x_test)

        # OLMM
        gp = GP(GP(OLMMKernel(m, p, obs_noise, 0.0, H, k1)))
        olmm = condition(gp, x_train, y_train_standardised)
        m_olmm = olmm.m(x_test)
        k_olmm = olmm.k(x_test)

        @test m_lmm ≈ m_olmm  atol = _TOL_
        @test k_lmm ≈ k_olmm  atol = _TOL_

        # LMM
        gp = GP(LMMKernel(Fixed(m), Fixed(p), Positive(obs_noise), Fixed(H), k2))
        lmm = condition(gp, x_train, y_train_standardised)
        m_lmm = lmm.m(x_test)
        k_lmm = lmm.k(x_test)

        # OLMM
        gp = GP(GP(OLMMKernel(m, p, obs_noise, 0.0, H, k2)))
        olmm = condition(gp, x_train, y_train_standardised)
        m_olmm = olmm.m(x_test)
        k_olmm = olmm.k(x_test)

        @test m_lmm ≈ m_olmm  atol = _TOL_
        @test k_lmm ≈ k_olmm  atol = _TOL_

        # obs_noise = 0.0

        # OLMM
        gp = GP(GP(OLMMKernel(m, p, 0.0, lat_noise, H, k1)))
        olmm = condition(gp, x_train, y_train_standardised)
        m_olmm = olmm.m(x_test)
        k_olmm = blocks(hourly_cov(olmm.k, x_test))

        # SOLMM
        y_train_transformed = (H \ y_train_standardised')'
        gp = GP(ZeroMean(), NoiseKernel(k1[1], lat_noise*DiagonalKernel()))
        K = gp.k(Observed(x_train))
        U = chol(K + GPForecasting._EPSILON_ .* Eye(K))
        k_ = gp.k(Latent(x_test), Latent(x_train))
        L_y = U' \ y_train_transformed
        k_U = k_ / U
        means_ = k_U * L_y
        vars_ = repmat(diag(gp.k(Observed(x_test)) - k_U * k_U'), 1, m)
        m_solmm = (H * means_')'
        k_solmm = []
        for i = 1:n
            push!(k_solmm, Matrix(Hermitian(H * Diagonal(vars_[i,:]) * H')))
        end

        @test m_solmm ≈ m_olmm  atol = _TOL_
        @test k_solmm ≈ k_olmm  atol = _TOL_

        # OLMM
        gp = GP(GP(OLMMKernel(m, p, 0.0, lat_noise, H, k2)))
        olmm = condition(gp, x_train, y_train_standardised)
        m_olmm = olmm.m(x_test)
        k_olmm = blocks(hourly_cov(olmm.k, x_test))

        # SOLMM
        y_train_transformed = (H \ y_train_standardised')'
        means_ = zeros(n, m)
        vars_ = zeros(n, m)
        for i = 1:m
            gp = GP(ZeroMean(), NoiseKernel(k2[i], lat_noise*DiagonalKernel()))
            solmm = condition(gp, Observed(x_train), y_train_transformed[:,i])
            means_[:,i] = solmm.m(Observed(x_test))
            vars_[:,i] = diag(solmm.k(Observed(x_test)))
        end
        m_solmm = (H * means_')'
        k_solmm = []
        for i = 1:n
            push!(k_solmm, Matrix(Hermitian(H * Diagonal(vars_[i,:]) * H')))
        end

        @test m_solmm ≈ m_olmm  atol = _TOL_
        @test k_solmm ≈ k_olmm  atol = _TOL_
    end
end
