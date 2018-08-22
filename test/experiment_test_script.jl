@everywhere function experiment_function(train_set_size, num_sample)
    # -- Define your experiment here, return the desired result to be saved, can be anything --#

    p = GP(EQ())
    x = collect(1:1:train_set_size)
    y = sample(p(x), num_sample)
    return y
end

"""
    set_parameters()

This function sets the parameters of your experiment.
The first dictionary, `parameters`, should be populated with `string => AbstractArray`
key-value pairs, where each `AbstractArray` contains the parameter values you would wish to
iterate over. The array could also only contain one element if you would like to use one
constant parameter.
The second dictionary should not be altered, it is used in the experiment execution. This
dictionary references the `experiment_function`, which should contain your experiment.
"""
function set_parameters()

    # -- Edit Experiment Parameters to be varied -- #
    parameters = [
        [24*7*10, 24*7*20],
        [2, 3],
    ]

    # -- Do not edit below this line -- #
    configuration = Dict{String, Any}(
        "parameters" => parameters,
        "experiment_function" => experiment_function,
        "seed" => 42,
        # "revision" => HelloBatch.getrevinfo(), # no revision test due to gitlab not having a .git folder during test time.
        # "date" => now(),
    )
    return configuration
end

# The script must return the configuration
return set_parameters()
