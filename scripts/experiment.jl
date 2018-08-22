using GPForecasting
# Parse command line arguments.
using ArgParse
args = GPForecasting.arg_parse(ARGS)

if isa(args["experiment-script"], Void) && isa(args["experiment"], Void)
    throw(UndefRefError(
        "You need to provide either an experiment name or an experiment script."
    ))
end
if !isa(args["experiment-script"], Void) && !isa(args["experiment"], Void)
    warn(
        """
        You provided both an experiment name and an experiment script.

        The code will proceed using the experiment script.
        """
    )
end
const configuration = isa(args["experiment-script"], Void) ?
    Symbol(args["experiment"]) : args["experiment-script"]

using HelloBatch
using AWSClusterManagers

global const AWS_RESOURCES = Dict{String, Union{String, Void}}()
AWS_RESOURCES["s3_uri"] = "s3://invenia-gpforecasting"


info("Experiment: $(args["experiment-script"])")
# If needed, add processes with the appropriate manager

cpus = args["nprocs"]
if cpus > 1
    info("Adding $(cpus - 1) processes")
    HelloBatch.unloadconda() do
        addprocs(HelloBatch.getmanager(args["cluster-manager"], cpus))
    end
    info("Now we have $(nprocs()) processes in total")
end

# Now that all the processes have been created, import in all of them the necessary packages
@everywhere using GPForecasting
@everywhere using HelloBatch
@everywhere import Memento: getlogger, setlevel!, DefaultHandler, DefaultFormatter, addlevel!
@everywhere addlevel!(getlogger(), "trace", 5) # Priority between debug and info

if isempty(args["archive"])
    info("Not archiving")
    output = GPForecasting.experiment(configuration)
else
    info("Archiving")

    # Define path for archiving
    epath = HelloBatch.experiment_path(args["label"])
    @everywhere mkpath($epath)

    #- The following part is for debugging purposes -#
    info("is_docker: ", HelloBatch.is_docker())

    println("Cluster Manager, container_id: ", AWSClusterManagers.container_id())
    try
        println("AWSBatchManager: ", AWSClusterManagers.AWSBatchManager())
    catch
        println("AWSClusterManagers.AWSBatchJob() failed")
    end
    #-----------------------------------------------#

    info("Destination: $epath")

    # Handler for logging into a file
    @everywhere push!(
        getlogger(),
        DefaultHandler(
            open(
                joinpath($epath, string("experiment.log.", myid())),
                "w",
            ),
            DefaultFormatter("[{date} | {name} | {level}] {msg} ({pid})"),
        ),
        # "logfile",
    )

    output = GPForecasting.experiment(configuration)

    results = Dict{String,Any}(
        "output" => output,
        "ENV" => ENV,
        "env" => split(readstring(`env`), "\n")
    )

    # Save the results
    HelloBatch.save_results(results, joinpath(epath, "results"))
    HelloBatch.save_results(args, joinpath(epath, "args"))

    # Archive in S3
    in("s3", args["archive"]) && HelloBatch.archive(epath, bucket=AWS_RESOURCES["s3_uri"])

end
