@testset "Node" begin
    n1 = GPForecasting.TreeNode([11], [])
    n2 = GPForecasting.TreeNode([22], [n1])
    n3 = GPForecasting.TreeNode([33], [])
    n4 = GPForecasting.TreeNode([44], [n2, n3])

    @test map(x -> x .+ 2, n2).x == [24]
    @test map(x -> x .+ 2, n2).children[1].x == [13]
    @test reduce(+, map(x -> x .+ 2, n4)) == [118]
    @test zip(n1, n3).x == ([11], [33])
    @test length(zip(n1, n3).children) == 0
    @test GPForecasting.flatten(n4) == [44, 22, 11, 33]
    @test GPForecasting.interpret(n2, [222, 333]).x == [222]
    @test GPForecasting.interpret(n2, [222, 333]).children[1].x == [333]
    @test isa(sprint(show, n1), String)
end
