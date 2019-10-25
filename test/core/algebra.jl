@testset "Algebra" begin
    @testset "Computations" begin
        A = rand(50, 50)
        B = rand(500, 500)
        C = rand(10, 10)

        @test GPForecasting.kron_lid_lmul(A, B) ≈ kron(Eye(10), A) * B atol = _ATOL_ rtol = _RTOL_
        @test GPForecasting.kron_rid_lmul_s(A, B) ≈ kron(A, Eye(10)) * B atol = _ATOL_ rtol = _RTOL_
        @test GPForecasting.kron_rid_lmul_m(A, B) ≈ kron(A, Eye(10)) * B atol = _ATOL_ rtol = _RTOL_
        @test GPForecasting.kron_lmul_rl(A, C, B) ≈ kron(A, C) * B atol = _ATOL_ rtol = _RTOL_
        @test GPForecasting.kron_lmul_lr(A, C, B) ≈ kron(A, C) * B atol = _ATOL_ rtol = _RTOL_
        @test GPForecasting.diag_outer_kron(A, C) ≈ diag(GPForecasting.outer(kron(A, C))) atol = _ATOL_ rtol = _RTOL_

        aa, bb, cc, dd = randn(5, 5), randn(10, 10), randn(5, 5), randn(10, 10)

        @test GPForecasting.diag_outer_kron(aa, bb, cc, dd) ≈
            diag(GPForecasting.outer(kron(aa, bb), kron(cc, dd))) atol = _ATOL_ rtol = _RTOL_
        @test GPForecasting.diag_outer(A) ≈ diag(GPForecasting.outer(A)) atol = _ATOL_ rtol = _RTOL_

        D = rand(50, 50)

        @test GPForecasting.diag_outer(A, D) ≈ diag(GPForecasting.outer(A, D)) atol = _ATOL_ rtol = _RTOL_

        B = LowerTriangular(B)

        @test GPForecasting.kron_lid_lmul_lt_s(A, B) ≈ GPForecasting.kron_lid_lmul(A, B) atol = _ATOL_ rtol = _RTOL_
        @test GPForecasting.kron_lid_lmul_lt_m(A, B) ≈ GPForecasting.kron_lid_lmul(A, B) atol = _ATOL_ rtol = _RTOL_

        U = Matrix{Float64}[]
        for i in 1:30
            push!(U, UpperTriangular(rand(100, 100)))
        end
        JJ = GPForecasting.Js(30)

        @test GPForecasting.sum_kron_J_ut(30, U...) ≈ sum([kron(U[i], JJ[i]) for i = 1:30]) atol = _ATOL_ rtol = _RTOL_

        V = [UpperTriangular(rand(10, 10)) for i in 1:5]
        MM = GPForecasting.Ms(5)
        A = rand(5, 5)

        @test UpperTriangular(GPForecasting.eye_sum_kron_M_ut(A, V...)) ≈
            UpperTriangular(Eye(5*10) + sum(
                [kron(V[i]*V[j]', A[i,j] * MM[i][j]) for i=1:5, j=1:5]
            )) atol = _ATOL_ rtol = _RTOL_
    end
end
