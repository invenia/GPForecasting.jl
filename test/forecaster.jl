@testset "Forecaster" begin
    # Most methods are not being tested here. Not sure what is the best way to test
    # things like fetch and fit without making the tests too slow.
    miso = mkt = Market{MISO}()
    gpf = GPForecaster(mkt, LMMKernel)
    @test Simulation.collection(gpf) == "miso"
    @test timezone(gpf) == timezone(mkt)
    sim_now = ZonedDateTime(2016, 6, 24, 9, tz"America/Winnipeg")
    @test isa(targets(gpf, sim_now), StepRange)
    @test isa(gpf.gp, GP)
    @test !gpf.standardise

    gpf = GPForecaster(mkt, OLMMKernel)
    @test Simulation.collection(gpf) == "miso"
    @test timezone(gpf) == timezone(mkt)
    sim_now = ZonedDateTime(2016, 6, 24, 9, tz"America/Winnipeg")
    @test isa(targets(gpf, sim_now), StepRange)
    @test isa(gpf.gp, GP)
    @test gpf.standardise
end
