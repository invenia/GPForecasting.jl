@testset "Neural Nets" begin
    @test relu(-5) == 0.0
    @test relu(5) == 5
    @test noisy_relu(50) > 0
    @test noisy_relu(5) != relu(5)
    @test leaky_relu(5) == relu(5)
    @test leaky_relu(-5) == -0.05
    @test softplus(0) == log(2)
    @test sigmoid(0) == 1/2

    W = rand(5, 3)
    B = rand(5)
    x = rand(3)
    l1 = NNLayer(W, B, Fixed(x -> x))
    @test l1(x) ≈ W * x + B
    l1 = NNLayer(W, B, Fixed(relu))
    @test l1(x) ≈ relu.(W * x + B)
    lb = BatchNormLayer(1, 0)
    @test mean(lb(x), dims=1) ≈ [0.0] atol = _ATOL_
    @test std(lb(x), dims=1) ≈ [1.0] atol = _ATOL_
    lb = BatchNormLayer(2, 5)
    @test mean(lb(x), dims=1) ≈ [5.0] atol = _ATOL_
    @test std(lb(x), dims=1) ≈ [2.0] atol = _ATOL_
    @test mean(lb(rand(3,3)), dims=1) ≈ fill(5.0, (1,3)) atol = _ATOL_
    @test std(lb(rand(3,3)), dims=1) ≈ fill(2.0, (1,3)) atol = _ATOL_
    nn = GPFNN([l1, lb])
    @test mean(lb(x)) ≈ 5.0 atol = _ATOL_
    @test std(lb(x)) ≈ 2.0 atol = _ATOL_
end
