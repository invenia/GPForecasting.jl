using ArgParse

"""
    argparse(cmd_args=ARGS) -> Dict{String, Any}

Parse the arguments passed to the scripts from the command line.
It uses ArgParse.jl to define

# Supported arguments

- 'label': specifies an identifier for the job, defaults to empty string;
- '--cluster-manager', '-c': specifies the ClusterManager to use ('local', 'docker', 'batch'),
defaults to 'local';
- '--nprocs', '-n': number of processes required, defaults to 1;
- '--archive', '-a':  save the job output to various locations (NB: array of strings produced);
- '--seed': specifies the seed for the random number generator, defaults to 42;
- '--experiment_script': path to the experiment to be run, defaults to "scripts/experiment_template.jl"

"""
function arg_parse(cmd_args=ARGS)
    parse_settings = ArgParseSettings()

    @add_arg_table parse_settings begin
        "label"
            help = "A label for identifying the job."
            default = ""
        "--cluster-manager", "-c"
            help = "Backend ClusterManager to use (e.g., 'local', 'docker', 'batch')."
            default = "local"
            range_tester = x -> in(x, ("local", "docker", "batch"))
        "--nprocs", "-n"
            help = "Number of processes requested."
            arg_type = Int
            default = 1
            range_tester = n -> n > 0
        "--archive", "-a"
            help = "Save the job output to various locations (e.g., 'local', 's3')."
            arg_type = AbstractString
            nargs = '*'
            range_tester = x -> in(x, ("local", "s3"))
        "--experiment-script", "-s"
            help = "Experiment script to be run." # We might want to pass different models to the experiment function
            arg_type = AbstractString
        "--experiment", "-e"
            help = "Experiment to run."
        "--seed"
            help = "Random number generator seed."
            arg_type = Int
            default = 42
    end

    return parse_args(cmd_args, parse_settings)
end
