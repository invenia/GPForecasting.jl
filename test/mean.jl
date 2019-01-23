@testset "Mean" begin
    x = collect(1:5)

    c = ConstantMean(5)
    @test c(x) == [5, 5, 5, 5, 5]
    @test (c * c)(x) == [5, 5, 5, 5, 5] .* [5, 5, 5, 5, 5]
    @test isa(sprint(show, c), String)

    fm = FunctionMean(x -> 2*x)
    @test fm(x) == 2 .* x
    @test (3 * fm)(x) == 6 .* x
    @test isa(sprint(show, fm), String)

    s = c + fm
    @test s(x) == [5, 5, 5, 5, 5] .+ 2 .* x
    @test isa(sprint(show, s), String)

    pos = PosteriorMean(ConstantKernel(), ConstantMean(), x, Eye(5), 2 .* x)
    @test pos(x) == 26.0 * ones(5)
    @test isa(sprint(show, pos), String)

    # Test Scaled Mean
    μ1 = ConstantMean(5)
    μ1 = 6 * μ1
    @test μ1(x) == [30.0, 30.0, 30.0, 30.0, 30.0]
    @test isa(sprint(show, μ1), String)

    # Test Product Mean
    μ2 = ConstantMean(8)
    μ3 = (μ2 * μ1)^2
    @test μ3(x) == [57600.0, 57600.0, 57600.0, 57600.0, 57600.0]
    @test isa(sprint(show, μ3), String)

    # Test multiplication by zero
    μ8 = 0.0 * μ1
    μ9 = μ8 * μ1

    # Test Zero Mean Properties
    @test isa(μ8, ZeroMean)
    @test isa(μ8 * μ1, ZeroMean)
    @test μ8 + μ1 == μ1
    @test μ1 + μ8 == μ1
    @test 3μ8 == μ8
    @test (3 + μ8)(x) ≈ ConstantMean(3.)(x) atol = _ATOL_
    @test (μ8 + 3)(x) ≈ ConstantMean(3.)(x) atol = _ATOL_
    @test isa(sprint(show, μ8), String)

    @test μ8(x) == [0.0, 0.0, 0.0, 0.0, 0.0]
    @test μ9(x) == [0.0, 0.0, 0.0, 0.0, 0.0]

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

    @testset "Mean Get/Set" begin
        y = GPForecasting.FunctionMean(sin)
        yy = 3.0 * ConstantMean()

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

        # Test Getting and setting
        test_grad = yy + yyy
        objj(a) = dot(test_grad(x), a)# Should return a scalar
        @test size(∇(objj)(rand(3))[1], 1) == 3 # Does not throw an error
        @test GPForecasting.get(test_grad) == [3.0]
        new_test_grad = GPForecasting.set(test_grad, [π])
        @test new_test_grad[:] ≈ [π] atol = _ATOL_# Test set
        @test new_test_grad(x) ≈ (π + 5.0)*ones(size(x, 1)) atol = _ATOL_

        # Test multiplication with bounded on top
        test_grad = (yy + yyy) * zz
        objj(a) = dot(test_grad(x), a)# Should return a scalar
        @test size(∇(objj)(rand(3))[1], 1) == 3
        @test GPForecasting.get(test_grad) ≈ [3.0, -14.346138220651198] atol = _ATOL_
        new_test_grad = GPForecasting.set(test_grad, [π, 10.0])
        @test new_test_grad[:] ≈ [π, 10.0] atol = _ATOL_
        @test GPForecasting.get(new_test_grad) ≈ [π, 10.0] atol = _ATOL_
        @test new_test_grad(x) ≈ ((π + 5.0) * 6.699925256613554)*ones(size(x, 1)) atol = 10 * _ATOL_

        # Test a posterior mean getting:
        p = GP(ConstantMean(Positive(3.0)), EQ())
        x = 0:0.1:1
        y = sin.(x)
        post = condition(p, x, y)
        @test GPForecasting.get(post.m)[1] ≈ 1.0986119553347207 atol = _ATOL_
    end
end
