@testset "Algebra" begin

    @testset "deprecated: BlockDiagonal" begin
        b1 = BlockDiagonal([rand(3, 3), rand(4, 4), rand(5, 5)])
        b2 = BlockDiagonal([rand(3, 2), rand(4, 4), rand(5, 3)])
        A = rand(size(b1)...)
        B = rand(size(b2)...)
        C = B'

        @test isa(sprint(show, b1), String)
        @test b1 ≈ b1
        @test b1 ≈ Matrix(b1)
        @test Matrix(b1) ≈ b1
        @test b1 * b1 ≈ Matrix(b1) * Matrix(b1)
        @test b1' * b1 ≈ Matrix(b1)' * Matrix(b1)
        @test b1 * b1' ≈ Matrix(b1) * Matrix(b1)'
        @test b1 * A ≈ Matrix(b1) * A
        @test b1 * A' ≈ Matrix(b1) * A'
        @test b1' * A ≈ Matrix(b1)' * A
        @test A * b1 ≈ A * Matrix(b1)
        @test A' * b1 ≈ A' * Matrix(b1)
        @test A * b1' ≈ A * Matrix(b1)'
        @test isa(b1 + Matrix(1.0I, size(b1)...), BlockDiagonal)
        @test diag(b1 + Matrix(1.0I, size(b1)...)) ≈ diag(b1) + ones(size(b1, 1)) atol = _ATOL_

        @test_throws DimensionMismatch b2 * b1
        @test_throws DimensionMismatch b2 * A
        @test_throws DimensionMismatch B * b1

        @test tr(b1) ≈ tr(Matrix(b1))

        b1 = BlockDiagonal(Hermitian.([(rand(3, 3) + 15Eye(3)), (rand(4, 4) + 15Eye(4)), (rand(5, 5) + 15Eye(5))]))
        Ub = cholesky(b1)
        Um = cholesky(Matrix(b1))
        @test Ub.L * Ub.U ≈ Matrix(b1) ≈ b1 ≈ Um.L * Um.U
        @test det(b1) ≈ det(Matrix(b1))
        @test eigvals(b1) ≈ eigvals(Matrix(b1))

        eqs = []
        for i in 1:size(b1, 1)
            for j in 1:size(b1, 2)
                push!(eqs, b1[i, j] ≈ Matrix(b1)[i, j])
            end
        end
        @test all(eqs)

        @test 5 * b1 ≈ 5 * Matrix(b1)
        @test b1 * 5 ≈ 5 * Matrix(b1)
        @test b1 / 5 ≈ Matrix(b1) / 5
        @test b1 + b1 ≈ 2 * b1
        @test isa(b1 + b1, BlockDiagonal)
    end

    @testset "Computations" begin
        A = rand(50, 50)
        B = rand(500, 500)
        C = rand(10, 10)

        @test GPForecasting.kron_lid_lmul(A, B) ≈ kron(Eye(10), A) * B atol = _ATOL_
        @test GPForecasting.kron_rid_lmul_s(A, B) ≈ kron(A, Eye(10)) * B atol = _ATOL_
        @test GPForecasting.kron_rid_lmul_m(A, B) ≈ kron(A, Eye(10)) * B atol = _ATOL_
        @test GPForecasting.kron_lmul_rl(A, C, B) ≈ kron(A, C) * B atol = _ATOL_
        @test GPForecasting.kron_lmul_lr(A, C, B) ≈ kron(A, C) * B atol = _ATOL_
        @test GPForecasting.diag_outer_kron(A, C) ≈ diag(GPForecasting.outer(kron(A, C))) atol = _ATOL_

        aa, bb, cc, dd = randn(5, 5), randn(10, 10), randn(5, 5), randn(10, 10)

        @test GPForecasting.diag_outer_kron(aa, bb, cc, dd) ≈
            diag(GPForecasting.outer(kron(aa, bb), kron(cc, dd))) atol = _ATOL_
        @test GPForecasting.diag_outer(A) ≈ diag(GPForecasting.outer(A)) atol = _ATOL_

        D = rand(50, 50)

        @test GPForecasting.diag_outer(A, D) ≈ diag(GPForecasting.outer(A, D)) atol = _ATOL_

        B = LowerTriangular(B)

        @test GPForecasting.kron_lid_lmul_lt_s(A, B) ≈ GPForecasting.kron_lid_lmul(A, B) atol = _ATOL_
        @test GPForecasting.kron_lid_lmul_lt_m(A, B) ≈ GPForecasting.kron_lid_lmul(A, B) atol = _ATOL_

        U = Matrix{Float64}[]
        for i in 1:30
            push!(U, UpperTriangular(rand(100, 100)))
        end
        JJ = GPForecasting.Js(30)

        @test GPForecasting.sum_kron_J_ut(30, U...) ≈ sum([kron(U[i], JJ[i]) for i = 1:30]) atol = _ATOL_

        V = [UpperTriangular(rand(10, 10)) for i in 1:5]
        MM = GPForecasting.Ms(5)
        A = rand(5, 5)

        @test UpperTriangular(GPForecasting.eye_sum_kron_M_ut(A, V...)) ≈
            UpperTriangular(Eye(5*10) + sum(
                [kron(V[i]*V[j]', A[i,j] * MM[i][j]) for i=1:5, j=1:5]
            )) atol = _ATOL_
    end

    @testset "deprecated: Cholesky decomposition of block diagonal matrices" begin
        X = [  4  12 -16
              12  37 -43
             -16 -43  98]
        U = [ 2.0 6.0 -8.0
              0.0 1.0  5.0
              0.0 0.0  3.0]
        B = BlockDiagonal([X, X])
        C = cholesky(B)
        @test C isa Cholesky{Float64, <:BlockDiagonal{Float64}}
        @test C.U ≈ BlockDiagonal([U, U])
        @test C.L ≈ BlockDiagonal([U', U'])
        @test C.UL ≈ C.U
        @test C.uplo === 'U'
        @test C.info == 0
        M = BlockDiagonal(map(Matrix, blocks(C.L)))
        C = Cholesky(M, 'L', 0)
        @test C.U ≈ BlockDiagonal([U, U])
        @test C.L ≈ BlockDiagonal([U', U'])
        @test C.UL ≈ C.L
        @test C.uplo === 'L'
        @test C.info == 0
    end
end
