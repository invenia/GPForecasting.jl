@testset "Synthetic" begin
    _RTOL_ = 1e-4

    function mse(means, y_true)
        return mean((y_true .- means).^2)
    end

    function logpdf(means, vars, y_true)
        return -0.5 * mean(log.(2π .* vars)) .- 0.5 * mean((y_true .- means).^2 ./ vars)
    end

    function gpforecasting_1d(m, k, x_train, y_train, x_test, y_test)
        gp = GP(m, k)
        b = condition(gp, Observed(x_train), y_train)
        pgp = learn(gp, Observed(x_train), y_train, mle_obj, its=50, trace=false)
        a = condition(pgp, Observed(x_train), y_train)
        return [
            mse(b.m(Latent(x_test)), y_test),
            logpdf(b.m(Latent(x_test)), diag(b.k(Latent(x_test))), y_test),
            mse(a.m(Latent(x_test)), y_test),
            logpdf(a.m(Latent(x_test)), diag(a.k(Latent(x_test))), y_test),
        ]
    end

    function gpforecasting_Nd(m, k, x_train, y_train)
        gp = GP(m, k)
        pgp = condition(gp, Observed(x_train), y_train)
        o = mle_obj(pgp, Observed(x_train), y_train)
        return [o(pgp.k[:]); ∇(o)(pgp.k[:])[1]]
    end

    # @testset "One dimensional problems" begin
    #     seed!(314159265)
    #     n = 50
    #     f_1d(x) = sin.(x) .* sin.(2.0 * x) + 0.25 * rand(n)
    #     x_train = sort(2pi .* rand(n))
    #     x_test = sort(2pi .* rand(n))
    #     y_train = f_1d(x_train)
    #     y_test = f_1d(x_test)
    #
    #     m = ZeroMean()
    #
    #     @testset "Kernels" begin
    #         k = NoiseKernel(2.0 * (EQ() ▷ 10.0), 0.001 * DiagonalKernel())
    #         @test gpforecasting_1d(m, k, x_train, y_train, x_test, y_test) ≈
    #             [0.14825538985254585, -1476.7191748817445,
    #             0.006339639909681351, 0.5729040630047109] rtol = _RTOL_
    #         k = NoiseKernel(2.0 * periodicise(EQ() ▷ 10.0, 2π), 0.001 * DiagonalKernel())
    #         @test gpforecasting_1d(m, k, x_train, y_train, x_test, y_test) ≈
    #             [0.13079765742455027, -760.2721834289244,
    #             0.010108277078091326, -0.3649906189846872] rtol = _RTOL_
    #         k = NoiseKernel(2.0 * (RQ(2.0) ▷ 5.0), 0.001 * DiagonalKernel())
    #         @test gpforecasting_1d(m, k, x_train, y_train, x_test, y_test) ≈
    #             [0.12226859947646133, -771.9774842426579,
    #             0.006339625546752299, 0.5728913634245152] rtol = _RTOL_
    #         k = NoiseKernel(2.0 * (MA(5/2) ▷ 10.0), 0.001 * DiagonalKernel())
    #         @test gpforecasting_1d(m, k, x_train, y_train, x_test, y_test) ≈
    #             [0.11075488248840884, -625.1263034476407,
    #             0.005838031744600918, 0.8934805590277801] rtol = _RTOL_
    #         k = NoiseKernel(2.0 * ((EQ() ▷ 10.0) + periodicise(EQ() ▷ 10.0, 2π) * (RQ(2.0) ▷ 10.0))
    #             + (MA(5/2) ▷ 10.0), 0.001 * DiagonalKernel())
    #         @test gpforecasting_1d(m, k, x_train, y_train, x_test, y_test) ≈
    #             [0.10434757275194265, -514.815138737102,
    #             0.005479951882127995, 0.3219275089452709] rtol = _RTOL_
    #     end
    #
    #     @testset "Parameters" begin
    #         k = NoiseKernel(2.0 * (EQ() ▷ Positive(10.0)), 0.001 * DiagonalKernel())
    #         @test gpforecasting_1d(m, k, x_train, y_train, x_test, y_test) ≈
    #             [0.14825538985254585, -1476.7191748817445,
    #             0.006339639909681351, 0.5729040630047109] rtol = _RTOL_
    #         k = NoiseKernel(2.0 * (EQ() ▷ Fixed(10.0)), 0.001 * DiagonalKernel())
    #         @test gpforecasting_1d(m, k, x_train, y_train, x_test, y_test) ≈
    #             [0.14825538985254585, -1476.7191748817445,
    #             0.15858203241161153, -9.226749455383052] rtol = _RTOL_
    #         k = NoiseKernel(2.0 * (EQ() ▷ Bounded(10.0, 9.0, 11.0)), 0.001 * DiagonalKernel())
    #         @test gpforecasting_1d(m, k, x_train, y_train, x_test, y_test) ≈
    #             [0.14825538985254585, -1476.7191748817445,
    #             0.15799166609928814, -9.14512057822587] rtol = _RTOL_
    #         k = NoiseKernel(Fixed(2.0) * (EQ() ▷ Positive(10.0) +
    #             periodicise(EQ() ▷ 10.0, Fixed(2π)) * RQ(Bounded(2.0, 1.5, 4.0)) ▷ 10.0) +
    #             MA(5/2) ▷ 10.0, 0.001 * DiagonalKernel())
    #         @test gpforecasting_1d(m, k, x_train, y_train, x_test, y_test) ≈
    #             [0.11769231376325562, -745.8368568684314,
    #             0.005934614644065813, 0.8602727131669052] rtol = _RTOL_
    #     end
    #
    #     @testset "Sampled from GP" begin
    #         k = NoiseKernel(EQ() ▷ 0.7, 0.01 * DiagonalKernel())
    #         p = condition(GP(m, k), Observed(x_train), y_train)
    #         seed!(314159265)
    #         y_sample = vec(sample(p(Observed(x_train))))
    #         k = NoiseKernel(EQ() ▷ 2.0, 0.02 * DiagonalKernel())
    #         pgp = learn(GP(m, k), Observed(x_train), y_sample, mle_obj, its=50, trace=false)
    #         @test exp.(pgp.k[:]) ≈ [0.7879293181025795, 0.005724751058727742] rtol = _RTOL_
    #     end
    # end
    #
    # @testset "Multidimensional input problems" begin
    #
    #     seed!(314159265)
    #     n = 100
    #     d = 3
    #     s = collect(1.0:1.0:d)
    #     f_Md(x) = prod(sin.(x .* s'), dims=2)[:] + 0.1 * rand(n)
    #     x_train = 2pi .* rand(n, d)
    #     y_train = f_Md(x_train)
    #
    #     m = ZeroMean()
    #
    #     k = NoiseKernel(1.0 * (EQ() ▷ [2.0 for i=1:d]), 0.001 * DiagonalKernel())
    #     @test gpforecasting_Nd(m, k, x_train, y_train) ≈
    #         [-563.8558154227734, -2.499981131718716e7, -534.6407925136859,
    #         -464.71299879973594, -404.09023778090716, -24779.550162742707] rtol = _RTOL_
    #     k = NoiseKernel(2.0 * periodicise(EQ() ▷ [5.0 for i=1:d],
    #         [2π for i=1:d]), 0.001 * DiagonalKernel())
    #     @test gpforecasting_Nd(m, k, x_train, y_train) ≈
    #         [-562.6989054719278, -4.9999872575423315e7, -232.66363721841196,
    #         -53.71585730176522, -177.59044544763736, -197.15399559966824,
    #         -190.27007463253395, -166.5338842075279, -24182.313430424398] rtol = _RTOL_
    #     k = NoiseKernel(2.0 * (RQ(Fixed(2.0)) ▷ [2.0 for i=1:d]), 0.001 * DiagonalKernel())
    #     @test gpforecasting_Nd(m, k, x_train, y_train) ≈
    #         [-564.1889437420803, -4.9999885050279684e7, -181.8608782523528,
    #         -159.26934605686552, -140.12542604317656, -24953.94896081927] rtol = _RTOL_
    #     k = NoiseKernel(2.0 * (MA(5/2) ▷ [2.0 for i=1:d]), 0.001 * DiagonalKernel())
    #     @test gpforecasting_Nd(m, k, x_train, y_train) ≈
    #         [-564.2224589732117, -4.999992177482256e7, -51.00221804594257,
    #         -52.366706897103924, -54.317938124597326, -24973.305713052014] rtol = _RTOL_
    # end

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
        mean_y_train = mean(y_train, dims=1)
        std_y_train = std(y_train, dims=1)
        y_train = (y_train .- mean_y_train) ./ std_y_train
        y_test = (y_test .- mean_y_train) ./ std_y_train

        H = Eye(p)
        k = [(EQ() ▷ 10.0) for i=1:5]
        gp = GP(LMMKernel(Fixed(5), Fixed(p), Positive(0.001), Fixed(H), k))
        b = condition(gp, x_train, y_train).m(x_test)
        gp = learn(gp, x_train, y_train, mle_obj, its=50, trace=false)
        a = condition(gp, x_train, y_train).m(x_test)
        @test mse(vec(b), vec(y_test)) ≈ 0.927596879880526  rtol = _RTOL_
        @test mse(vec(a), vec(y_test)) ≈ 0.8087681056630357 rtol = _RTOL_

        U, S, V = svd(cov(y_train))
        H = U * Diagonal(sqrt.(S))
        k = [(EQ() ▷ 10.0) for i=1:5]
        gp = GP(LMMKernel(Fixed(5), Fixed(p), Positive(0.001), Fixed(H), k))
        b = condition(gp, x_train, y_train).m(x_test)
        gp = learn(gp, x_train, y_train, mle_obj, its=50, trace=false)
        a = condition(gp, x_train, y_train).m(x_test)
        @test mse(vec(b), vec(y_test)) ≈ 0.8751756874359901  rtol = _RTOL_
        @test mse(vec(a), vec(y_test)) ≈ 0.10721174421720554 rtol = _RTOL_

        H = (U * Diagonal(sqrt.(S)))[:,1:3]
        k = [(EQ() ▷ 10.0) for i=1:3]
        gp = GP(LMMKernel(Fixed(3), Fixed(p), Positive(0.001), Fixed(H), k))
        b = condition(gp, x_train, y_train).m(x_test)
        gp = learn(gp, x_train, y_train, mle_obj, its=50, trace=false)
        a = condition(gp, x_train, y_train).m(x_test)
        @test mse(vec(b), vec(y_test)) ≈ 0.8754142897078707  rtol = _RTOL_
        @test mse(vec(a), vec(y_test)) ≈ 0.10636472286495949 rtol = _RTOL_

        H = (U * Diagonal(sqrt.(S)))[:,1:1]
        k = [(EQ() ▷ 10.0)]
        gp = GP(LMMKernel(Fixed(1), Fixed(p), Positive(0.001), Fixed(H), k))
        b = condition(gp, x_train, y_train).m(x_test)
        gp = learn(gp, x_train, y_train, mle_obj, its=50, trace=false)
        a = condition(gp, x_train, y_train).m(x_test)
        @test mse(vec(b), vec(y_test)) ≈ 0.8846475452351197  rtol = _RTOL_
        @test mse(vec(a), vec(y_test)) ≈ 0.11874106402148683 rtol = _RTOL_

        seed!(314159265)
        H = rand(p, 7)
        k = [(EQ() ▷ 10.0) for i=1:7]
        gp = GP(LMMKernel(Fixed(7), Fixed(p), Positive(0.001), Fixed(H), k))
        b = condition(gp, x_train, y_train).m(x_test)
        gp = learn(gp, x_train, y_train, mle_obj, its=50, trace=false)
        a = condition(gp, x_train, y_train).m(x_test)
        @test mse(vec(b), vec(y_test)) ≈ 0.8202143612660611  rtol = _RTOL_
        @test mse(vec(a), vec(y_test)) ≈ 0.20612005452907958 rtol = _RTOL_
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
        mean_y_train = mean(y_train, dims=1)
        std_y_train = std(y_train, dims=1)

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
        gp = GP(OLMMKernel(m, p, obs_noise, 0.0, H, k1))
        olmm = condition(gp, x_train, y_train_standardised)
        m_olmm = olmm.m(x_test)
        k_olmm = olmm.k(x_test)

        @test m_lmm ≈ m_olmm  rtol = _RTOL_
        @test k_lmm ≈ k_olmm  rtol = _RTOL_

        # LMM
        gp = GP(LMMKernel(Fixed(m), Fixed(p), Positive(obs_noise), Fixed(H), k2))
        lmm = condition(gp, x_train, y_train_standardised)
        m_lmm = lmm.m(x_test)
        k_lmm = lmm.k(x_test)

        # OLMM
        gp = GP(OLMMKernel(m, p, obs_noise, 0.0, H, k2))
        olmm = condition(gp, x_train, y_train_standardised)
        m_olmm = olmm.m(x_test)
        k_olmm = olmm.k(x_test)

        @test m_lmm ≈ m_olmm  rtol = _RTOL_
        @test k_lmm ≈ k_olmm  rtol = _RTOL_

        # obs_noise = 0.0

        # OLMM
        gp = GP(OLMMKernel(m, p, 0.0, lat_noise, H, k1))
        olmm = condition(gp, x_train, y_train_standardised)
        m_olmm = olmm.m(x_test)
        k_olmm = blocks(hourly_cov(olmm.k, x_test))

        # SOLMM
        y_train_transformed = (H \ y_train_standardised')'
        gp = GP(ZeroMean(), NoiseKernel(k1[1], lat_noise*DiagonalKernel()))
        K = gp.k(Observed(x_train))
        U = cholesky(K + GPForecasting._EPSILON_ .* Eye(K)).U
        k_ = gp.k(Latent(x_test), Latent(x_train))
        L_y = U' \ y_train_transformed
        k_U = k_ / U
        means_ = k_U * L_y
        vars_ = repeat(diag(gp.k(Observed(x_test)) - k_U * k_U'), 1, m)
        m_solmm = (H * means_')'
        k_solmm = []
        for i = 1:n
            push!(k_solmm, Matrix(Hermitian(H * Diagonal(vars_[i,:]) * H')))
        end

        @test m_solmm ≈ m_olmm  rtol = _RTOL_
        @test k_solmm ≈ k_olmm  rtol = _RTOL_

        # OLMM
        gp = GP(OLMMKernel(m, p, 0.0, lat_noise, H, k2))
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

        @test m_solmm ≈ m_olmm  rtol = _RTOL_
        @test k_solmm ≈ k_olmm  rtol = _RTOL_
    end

    @testset "Sparse GPs" begin
        f = open("sgp_train_inputs")
        s = read(f, String);
        x_train = parse.(Float64, split(s, "\n")[1:end-1]);
        close(f)

        f = open("sgp_train_outputs")
        s = read(f, String);
        y_train = parse.(Float64, split(s, "\n")[1:end-1]);
        close(f)

        gp = GP(1.0 * (EQ() ▷ 0.5) + 1e-2 * DiagonalKernel())
        ngp = learn(gp, x_train, y_train, mle_obj, its = 50, trace = false)
        pos = condition(ngp, x_train, y_train)

        x_test = collect(0:0.01:6);
        means, lb, ub = credible_interval(pos, x_test)

        # Let's force this to be a DataFrame just make the test more thorough
        xtrain = DataFrame([x_train, rand(size(x_train, 1))], [:data, :bs])
        Xm_i = xtrain[rand(1:size(xtrain, 1), 15), :]

        sgp, Xm, σ² = GPForecasting.learn_sparse(
            GP((1.0 * (EQ() ▷ 0.5)) ← :data),
            xtrain,
            y_train,
            Xm_i,
            Positive(0.1),
            its = 50,
            trace = false
        )
        spos = GPForecasting.condition_sparse(sgp, xtrain, Xm, y_train, σ²)

        xtest = DataFrame([x_test, rand(size(x_test, 1))], [:data, :bs])
        smeans, slb, sub = credible_interval(spos, Observed(xtest));

        @test mean(abs.(means .- smeans)) / mean(abs.(means)) < 0.01
        @test mean(abs.(lb .- slb)) / mean(abs.(lb)) < 0.01
        @test mean(abs.(ub .- sub)) / mean(abs.(ub)) < 0.01
        @test isa(Xm, DataFrame)
        @test Xm_i[:, :bs] ≈ Xm[:, :bs]
        @test !(Xm[:, :data] ≈ Xm_i[:, :data])
        @test size(spos.k(Observed(xtest[1:5, :]), Observed(xtest[1:8, :]))) == (5, 8)
        @test size(spos.k(Latent(xtest[1:5, :]), Observed(xtest[1:8, :]))) == (5, 8)
        @test size(spos.k(xtest[1:5, :], xtest[1:8, :])) == (10, 16)
        smeansl, slbl, subl = credible_interval(spos, Latent(xtest));
        @test smeansl ≈ smeans atol = _ATOL_ rtol = _RTOL_
        @test all(slbl .> slb)
        @test all(subl .< sub)
        @test size(spos.m(xtest[1:5, :])) == (5, 2)
        @test size(spos.m(Observed(xtest[1:5, :]))) == (5,)

        # OLMM
        ytrain = [y_train y_train]
        H = [1.0 0.0; 0.0 1.0]
        k = 1.0 * (EQ() ▷ 0.5) + Fixed(1e-2) * DiagonalKernel()
        gp = GP(OLMMKernel(
            Fixed(2),
            Fixed(2),
            Fixed(1e-2),
            Fixed(1e-2),
            Fixed(H),
            Fixed(H),
            Fixed(H),
            Fixed([1.0, 1.0]),
            [k for i in 1:2])
        )
        pos = condition(gp, x_train, ytrain)
        means, lb, ub = credible_interval(pos, x_test)

        k = (1.0 * (EQ() ▷ 0.5)) ← :data
        sgp = GP(OLMMKernel(
            Fixed(2),
            Fixed(2),
            Fixed(1e-2),
            Fixed(1e-2),
            Fixed(H),
            Fixed(H),
            Fixed(H),
            Fixed([1.0, 1.0]),
            [k for i in 1:2])
        )
        spos = GPForecasting.condition_sparse(sgp, xtrain, xtrain, ytrain, 0.01)
        smeans, s_lb, s_ub = credible_interval(spos, Observed(xtest))
        @test mean(abs.(means .- smeans)) / mean(abs.(means)) < 0.01
        @test mean(abs.(lb .- s_lb)) / mean(abs.(lb)) < 0.01
        @test mean(abs.(ub .- s_ub)) / mean(abs.(ub)) < 0.01
    end
end
