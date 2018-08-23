"""
    get_parameters(configuration::String, options::Associative=Dict()) -> Dict

Retrieves the parameter for the experiment specified by `configuration`, overriding the
default settings as specified in `options`. See `update_exper_def!` for the mechanics.

# Input
- `configuration::String`: the name of the model that will be used
- `options::Associative=Dict()`: a dictionary with options, if needed, to override the
defaults

# Output
- A Dict with three keys
    - `parameters`: a vector of vectors containing all options
    - `experiment function`: function to execute the experiment
    - `seed`: seed, for reproducibility
    - `revision`: revision information
    - `date`: date in which the experiment was ran
"""
function get_parameters(configuration::Symbol, options::Associative=Dict())

    # Look for the case
    try
        global param_function = Symbol(configuration)
    catch e
        throw(ArgumentError("Unknown experiment specification. $e"))
    end
    # Create the dict
    try
        global exper_params = eval(param_function)()
    catch e
        throw(ArgumentError("Unknown param_function. $e"))
    end
    # Update the parameters, if needed.
    length(options) > 0 && update_exper!(exper_params, options)

    return exper_params
end

"""

    get_parameters(experiment_path::AbstractString) -> Dict

Retrieves the parameter for the experiment specified by `experiment_path`.

# Input
- `experiment_path::AbstractString`: The path of the experiment.

# Output
- A Dict with three keys
    - `parameters`: a vector of Dict containing the numerical coefficients for the tunable
    parameters of the experiment
    - `experiment_function`: the function which to call to execute the experiment
    - `seed`: the experiment seed
"""
function get_parameters(experiment_path::AbstractString)
    include(experiment_path)
end

"""
    describe(configuration::Union{String, Symbol})

Description of a given experiment `configuration`.
"""
function describe(configuration::Symbol)
    f = Symbol("describe_" * String(configuration))
    eval(f)()
end

"""
    source(configuration::Union{String, Symbol})

Source code location for a given experiment `configuration`.
"""
function source(configuration::Symbol)
    f = Symbol("source_" * String(configuration))
    eval(f)()
end

"""
    update_exper!(exper::Associative, options::Associative)

Update the `exper` dictionary with the keys found in `options`.

# Arguments

- `exper::Associative`: the dictionary containing the specifications of the experiment;
- `options::Associative`: the dictionary containing the changes that the user wants to make
on the default settings.
"""
function update_exper!(exper::Associative, options::Associative)
    # This will be needed for the check of unused options
    expected_keys = String[]

    # For each entry type, execute the update
    for k in keys(exper)
        haskey(options, k) && update!(k, options[k], exper)
        push!(expected_keys, k)
    end

    diff = setdiff(keys(options), expected_keys)
    !isempty(diff) &&
        throw(ArgumentError(logger, "Unknown options, $diff, please check"))

    return exper
end

function update!(k::AbstractString, v::Any, exper::Associative)
    exper[k] = v
end