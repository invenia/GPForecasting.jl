@testset "Mean" begin
    x = collect(1:5)

    c = ConstantMean(5)
    @test c(x) == [5, 5, 5, 5, 5]
    @test (c * c)(x) == [5, 5, 5, 5, 5] .* [5, 5, 5, 5, 5]

    fm = FunctionMean(x -> 2*x)
    @test fm(x) == 2 .* x
    @test (3 * fm)(x) == 6 .* x

    s = c + fm
    @test s(x) == [5, 5, 5, 5, 5] .+ 2 .* x

    pos = PosteriorMean(ConstantKernel(), ConstantMean(), x,  eye(5), 2 .* x)
    @test pos(x) == 26.0 * ones(5)

    # Test Scaled Mean
    μ1 = ConstantMean(5)
    μ1 = 6 * μ1
    @test μ1(x) == [30.0, 30.0, 30.0, 30.0, 30.0]

    # Test Product Mean
    μ2 = ConstantMean(8)
    μ3 = (μ2 * μ1)^2
    @test μ3(x) == [57600.0, 57600.0, 57600.0, 57600.0, 57600.0]

    # Test Division Mean
    μ4 = μ1 / μ2
    μ5 = μ2 / μ1
    μ6 = 8  / μ1
    μ7 = μ2 / 30
    @test μ4(x) == [3.75, 3.75, 3.75, 3.75, 3.75]
    @test μ5(x) ≈ [0.266667, 0.266667, 0.266667, 0.266667, 0.266667] atol = 1e-6
    @test μ6(x) ≈ μ5(x) atol = 1e-8
    @test μ7(x) ≈ μ5(x) atol = 1e-8

    # Test multiplication/division by zero
    μ8 = 0.0 * μ6
    μ9 = μ8 * μ6
    μ10 = μ8 / μ6

    # Test Zero Mean Properties
    @test isa(μ8, ZeroMean)
    @test isa(μ8 * μ1, ZeroMean)
    @test μ8 + μ1 == μ1
    @test μ1 + μ8 == μ1
    @test 3μ8 == μ8
    @test (3 + μ8)(x) ≈ ConstantMean(3.)(x) atol = _ATOL_
    @test (μ8 + 3)(x) ≈ ConstantMean(3.)(x) atol = _ATOL_
    @test (μ8 - 3)(x) ≈ ConstantMean(-3.)(x) atol = _ATOL_
    @test (3 - μ8)(x) ≈ ConstantMean(3.)(x) atol = _ATOL_
    @test (3 ^ μ8)(x) ≈ ConstantMean(1.)(x) atol = _ATOL_
    @test (μ8 ^ 3)(x) ≈ ConstantMean(0.0)(x) atol = _ATOL_

    # @test_throws ErrorException μ11 = μ6 / μ8
    # Altered to the default julia behaviour: 1/0 = Inf
    # We can change this back to throwing an error if you'd like
    @test (μ6 / μ8)(x) == [Inf, Inf, Inf, Inf, Inf]

    @test μ8(x) == [0.0, 0.0, 0.0, 0.0, 0.0]
    @test μ9(x) == [0.0, 0.0, 0.0, 0.0, 0.0]
    @test μ10(x) == [0.0, 0.0, 0.0, 0.0, 0.0]

    # Test sum of all types multiplied
    μ∞ = (0.001μ1 + 1e-2μ2 + 1e-6μ3 + 1e-1μ4)^2
    @test μ∞(x) ≈ [0.294415, 0.294415, 0.294415, 0.294415, 0.294415] atol = 1e-6

    # Test exponentiaion
    x = 0:0.1:1
    μ₁ = ConstantMean(Positive(5.0)) ^ (-1.0)
    μ₂ = FunctionMean(sin) ^ (-1.0)
    μ₃ = μ₁ + μ₂ + (3.0 ^ ScaledMean(2.0))
    @test μ₁(x) ≈ (5.0 ^ (-1.0))*ones(size(x, 1)) atol = _ATOL_
    @test μ₂(x) ≈ (sin.(x)).^(-1.0) atol = _ATOL_
    @test μ₃(x) ≈ μ₁(x) + μ₂(x) + (3.0 ^ ScaledMean(2.0))(x) atol = _ATOL_
    @test ConstantMean(2.0)(x) ≈ ScaledMean(2.0)(x) atol = _ATOL_

    # Test Matrix Multiplication
    x = 0:1:5
    A = EQ()([1.0, 2.0], [1.0, 2.0])
    m1 = ConstantMean()
    pm = GPForecasting.PosteriorMean(EQ(), ConstantMean(), [1, 2, 3], rand(3,3), [1, 2, 3])
    ξ1 = A[1, 1] * pm + A[1, 2] * m1
    ξ2 = A[2, 1] * pm + A[2, 2] * ConstantMean()
    @test (A * [pm, m1])(x) == hcat(ξ1(x), ξ2(x))
    @test ([3.5, 4.6] * pm)(x) == hcat(3.5pm(x), 4.6pm(x))
    @test ([3.5, 4.6] .* pm)(x) == hcat(3.5pm(x), 4.6pm(x))
    @test ([3.5, 4.6] + pm)(x) == hcat((3.5 + pm)(x), (4.6 + pm)(x))
    @test ([3.5, 4.6] - pm)(x) == hcat((3.5 - pm)(x), (4.6 - pm)(x))
    @test ([3.5, 4.6] / pm)(x) == hcat((3.5 / pm)(x), (4.6 / pm)(x))

    @testset "Mean Get/Set" begin
        y = GPForecasting.FunctionMean(sin)
        yy = GPForecasting.ConstantMean(3.0)

        yyy = GPForecasting.ConstantMean(Fixed(5.0))
        z = 5.0 * ConstantMean(Positive(5.0))
        zz = ConstantMean(Bounded(4.0, 5.0, 6.7))
        zzz = ConstantMean(Named(5.0, "Five"))
        zzzz = ConstantMean(DynamicBound(x -> x, [5.0]))
        params = [yyy, z, zz, zzz]

        testt = yy * y

        # Test multiplication
        x = [1.0, 2.0, 3.0]
        @test testt(x) == yy(x) .* y(x) # Returns true

        # Test Division
        testte = yy / y
        @test testte(x) == yy(x) ./ y(x) # Returns true

        # Test
        difficult = 30 * yy / π * testt
        @test difficult(x) == 30 * yy(x) ./ π .* testt(x)  # Returns true

        # Test
        difficult3 = (y * y) / (30 * yy * yy) / π * testt
        @test difficult3(x) == (y(x) .* y(x)) ./ (30 .* yy(x) .* yy(x)) ./ π .* testt(x) # Returns true

        # Test Getting and setting on very hard combination
        final = testte * difficult - difficult3
        @test final(x) == testte(x) .* difficult(x) .- difficult3(x)
        @test GPForecasting.get(final) == [GPForecasting.get(testte); GPForecasting.get(difficult); GPForecasting.get(difficult3)]

        # Test Getting and setting
        test_grad = yy + yyy
        objj(a) = dot(test_grad(x), a)# Should return a scalar
        @test size(∇(objj)(rand(3))[1], 1) == 3 # Does not throw an error
        @test GPForecasting.get(test_grad) == [3.0]
        new_test_grad = GPForecasting.set(test_grad, [π])
        @test new_test_grad[:] ≈ [π] atol = _ATOL_# Test set
        @test new_test_grad(x) ≈ (π + 5.0)*ones(size(x, 1)) atol = _ATOL_

        # Test subtraction with positive
        test_grad = yy + yyy - z
        objj(a) = dot(test_grad(x), a)# Should return a scalar
        @test size(∇(objj)(rand(3))[1], 1) == 3
        @test GPForecasting.get(test_grad) ≈ [3.0, 5.0, 1.60944] atol = _ATOL_
        new_test_grad = GPForecasting.set(test_grad, [π, 3.0, 2.0])
        @test new_test_grad[:] ≈ [π, 3.0, 2.0] atol = _ATOL_
        @test GPForecasting.get(new_test_grad) ≈ [π, 3.0, 2.0] atol = _ATOL_
        @test new_test_grad(x) ≈ (π + 5.0 - 3.0 * exp(2.0))*ones(size(x, 1)) atol = _ATOL_

        # Test multiplication with bounded on top
        test_grad = (yy + yyy - z) * zz
        objj(a) = dot(test_grad(x), a)# Should return a scalar
        @test size(∇(objj)(rand(3))[1], 1) == 3
        @test GPForecasting.get(test_grad) ≈ [3.0, 5.0, 1.6094377124340804, -14.346138220651198] atol = _ATOL_
        new_test_grad = GPForecasting.set(test_grad, [π, 3.0, 2.0, 10.0])
        @test new_test_grad[:] ≈ [π, 3.0, 2.0, 10.0] atol = _ATOL_
        @test GPForecasting.get(new_test_grad) ≈ [π, 3.0, 2.0, 10.0] atol = _ATOL_
        @test new_test_grad(x) ≈ ((π + 5.0 - 3.0 * exp(2.0)) * 6.699925256613554)*ones(size(x, 1)) atol = _ATOL_

        # Test division with named on the bottom
        test_grad = (yy + yyy - z) * zz / zzz
        objj(a) = dot(test_grad(x), a)# Should return a scalar
        ∇(objj)(rand(3)) # Does not throw an error
        @test GPForecasting.get(test_grad) ≈ [3.0, 5.0, 1.6094377124340804, -14.346138220651198, 5.0] atol = _ATOL_
        new_test_grad = GPForecasting.set(test_grad, [π, 3.0, 2.0, 10.0, 6.7])
        @test new_test_grad[:] ≈ [π, 3.0, 2.0, 10.0, 6.7] atol = _ATOL_
        @test GPForecasting.get(new_test_grad) ≈ [π, 3.0, 2.0, 10.0, 6.7] atol = _ATOL_
        @test new_test_grad(x) ≈ ((π + 5.0 - 3.0 * exp(2.0)) * 6.699925256613554 / 6.7)*ones(size(x, 1)) atol = _ATOL_

        # Test a posterior mean getting:
        p = GP(ConstantMean(Positive(3.0)), EQ())
        x = 0:0.1:1
        y = sin.(x)
        post = condition(p, x, y)
        @test GPForecasting.get(post.m)[1] ≈ 1.0986119553347207 atol = _ATOL_
    end
end
