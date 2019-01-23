@testset "Parameter" begin
    @test GPForecasting.pack([1, 2, 3]) == [1, 2, 3]
    @test GPForecasting.unpack(Matrix(undef, 2, 2), [1, 2, 3, 4]) == [1 3; 2 4]
    @test GPForecasting.set([1], [1, 2, 3]) == [1, 2, 3]

    fx = Fixed(5.5)

    @test fx ≈ fx
    @test GPForecasting.unwrap(fx) ≈ 5.5 atol = _ATOL_
    @test GPForecasting.pack(fx) == Float64[]
    @test GPForecasting.unpack(fx, [0]) ≈ fx
    @test isa(GPForecasting.name(fx), Nullable{String})
    @test GPForecasting.set(fx, 12.3) ≈ Fixed(12.3)
    @test isa(sprint(show, fx), String)

    v = [5.5, 7.7]
    fv = Fixed(v)

    @test fv ≈ fv
    @test GPForecasting.unwrap(fv) ≈ v atol = _ATOL_
    @test GPForecasting.pack(fv) == Float64[]
    @test GPForecasting.unpack(fv, []) ≈ fv
    @test isa(GPForecasting.name(fv), Nullable{String})
    @test GPForecasting.set(fv, [1.1, 3.3]) ≈ Fixed([1.1, 3.3])
    @test isa(sprint(show, fv), String)

    p = Positive(5.5)

    @test p ≈ p
    @test GPForecasting.unwrap(p) ≈ 5.5 atol = _ATOL_
    @test GPForecasting.pack(p) == Float64[log(5.5 - p.ε)]
    @test GPForecasting.unpack(p, [log(5.5 - p.ε)]) ≈ p
    @test isa(GPForecasting.name(p), Nullable{String})
    @test GPForecasting.set(p, 12.3) ≈ Positive(12.3)
    @test isa(sprint(show, p), String)

    v = [-1.0, 1.0]
    pv = Positive(v)

    @test pv ≈ pv
    @test GPForecasting.unwrap(pv) ≈ v atol = _ATOL_
    @test GPForecasting.pack(pv) ≈ [-13.815510557964274, 1.0e-6] atol = _ATOL_
    @test GPForecasting.unpack(pv, GPForecasting.pack(pv)).p ≈ [2.0e-6, 1.0] atol = _ATOL_
    @test isa(GPForecasting.name(p), Nullable{String})
    @test GPForecasting.set(pv, [0.5, 1.5]) ≈ Positive([0.5, 1.5])
    @test isa(sprint(show, pv), String)

    n = Named(5.5, "secret")

    @test n ≈ n
    @test GPForecasting.unwrap(n) ≈ 5.5 atol = _ATOL_
    @test GPForecasting.pack(n) == Float64[5.5]
    @test GPForecasting.unpack(n, [5.5]) ≈ n
    @test GPForecasting.name(n) == "secret"
    @test GPForecasting.set(n, 12.3) ≈ Named(12.3, "secret")
    @test isa(sprint(show, n), String)

    b = Bounded(0., -10., 10.)

    @test b ≈ b
    @test GPForecasting.unwrap(b) ≈ 0. atol = _ATOL_
    @test isa(GPForecasting.name(b), Nullable{String})
    @test GPForecasting.set(b, 5.) ≈ Bounded(5., -10., 10.)
    @test isa(sprint(show, b), String)
    for x in range(0., stop=10., length=10)
        b = Bounded(x, 10.)
        @test GPForecasting.unwrap(GPForecasting.unpack(b, GPForecasting.pack(b))) ≈ x  atol = _ATOL_
    end
    b = Bounded(-1., 10.)
    @test GPForecasting.unwrap(GPForecasting.unpack(b, GPForecasting.pack(b))) ≈ 0.  atol = _ATOL_
    b = Bounded(11., 10.)
    @test GPForecasting.unwrap(GPForecasting.unpack(b, GPForecasting.pack(b))) ≈ 10.  atol = _ATOL_

    v = [3.0, 7.0]
    lb = [4.0, 1.0]
    ub = [6.0, 10.0]
    bv = Bounded(v, lb, ub)

    @test bv ≈ bv
    @test GPForecasting.unwrap(bv) ≈ v atol = _ATOL_
    @test GPForecasting.pack(bv) ≈ [-14.508657238384316, 0.6931470138932831] atol = _ATOL_
    @test GPForecasting.unpack(bv, GPForecasting.pack(bv)).p ≈ [4.0, 7.0] atol = _ATOL_
    @test isa(GPForecasting.name(bv), Nullable{String})
    @test GPForecasting.set(bv, [5.0, 2.0]) ≈ Bounded([5.0, 2.0], lb, ub)
    @test isa(sprint(show, bv), String)
end
