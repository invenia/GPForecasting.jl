using IterTools

"""
    function experiment(
        exp_config::Dict,
        seed::Int=42,
    ) -> Dict

Run the experiment with the specified configuration.

# Inputs

- `exp_config::Dict`: dictionary outlining the experiment configuration
- `seed::Int=42`: the seed for the RNG.
"""
function experiment(
    exp_config::Dict;
    seed::Int=42,
    trace::Bool = true,
)

    parameters = exp_config["parameters"]
    exp_fn = exp_config["experiment_function"]
    try srand(exp_config["seed"]) catch srand(seed) end # Set the seed for pseudorandomness

    results = pmap(
                product(parameters...),
                on_error = ex -> (info(GPForecasting.LOGGER, string(ex)); string(ex)),
                ) do param

        if trace info(LOGGER, "Testing Parameter Configuration: $param") end
        tic()
        res = Dict{String, Any}(
            "parameters" => param,
            "output" => exp_fn(param...),
            "time" => if trace toc() else toq() end,
        )
        return res
    end

    return results
end


"""
    function experiment(
        exp_path::AbstractString,
        seed::Int=42,
    ) -> Dict

Run the experiment specified in the script.

# Inputs

- `exp_path::Dict`: path to script outlining the experiment.
- `seed::Int=42`: the seed for the RNG.
"""
function experiment(
    exp_path::Union{AbstractString, Symbol};
    seed::Int=42,
    trace::Bool = true,
)
    pars = GPForecasting.get_parameters(exp_path)
    return experiment(pars, seed = seed, trace = trace)
end
