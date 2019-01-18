module Experiments

export list_experiments, describe, source, get_parameters

using GPForecasting
using Compat: AbstractDict, repeat
using Compat.Dates
using DataFrames
using Nabla
using HelloBatch
using ModelAnalysis
using Missings

include("experiment_template.jl")
include("basicLMM.jl")
include("basicOLMM.jl")
include("basicIOLMM.jl")
include("hour_loadLMM.jl")
include("damec.jl")
# include("indep.jl")
include("initialisedLMM.jl")
include("utils.jl")
include("comparePJM.jl")

"""
    list_experiments()

List of all experiments available. Use `describe(experiment)` for the description of a given
`experiment`.
"""
function list_experiments()
    # NOTE: be sure to update this every time you create a new experiment. This list should
    # be kept up-to-date.
    return [
        experiment_template,
        basicLMM,
        basicOLMM,
        basicIOLMM,
        hour_loadLMM,
        # indep,
        initialisedLMM,
        damec,
        comparePJM,
    ]
end

end # module end
