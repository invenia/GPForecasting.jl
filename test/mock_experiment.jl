# Mock custom experiment

options = Dict{String, Any}(
    "parameters" => [
        [24*7*1, 24*7*2],
        [4, 5],
    ]
)

function mock_params(exp::Symbol, opt::Dict)
    return GPForecasting.Experiments.get_parameters(exp, opt)
end

mock_params(:experiment_template, options)
