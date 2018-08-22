# Using GPForecasting with HelloBatch

## Quick way

0. Clone this repository (and checkout this branch)

1. Write your experiment in the `experiment()` function of the `scr/experiment.jl` file. At the end of the function, simply `return` your experiment result.

2. Start your docker engine locally (`sudo service docker start`) or manually click on the docker icon on your computer and wait for it to load.

3. Set up your AWS profile:
    - Install [aws-cli](https://github.com/aws/aws-cli).
    - In your `~/.aws/config` file (if it does not exist, create it), paste the following credentials:

```
[profile GPForecasting@Invenia]
region = us-east-1
source_profile = Administrator@Invenia
role_arn = arn:aws:iam::052722095006:role/Researchers
s3 =
  max_concurrent_requests = 20
  max_queue_size = 10000
  multipart_threshold = 64MB
  multipart_chunksize = 16MB
  addressing_style = virtual
  signature_version = s3v4
```
    This will give you access to the `GPForecasting@Invenia` account.

4. Pull down the Docker parent image "julia-baked:0.6"

```
export AWS_PROFILE='GPForecasting@Invenia'
$(aws ecr get-login --no-include-email --registry-ids 292522074875)
docker pull 292522074875.dkr.ecr.us-east-1.amazonaws.com/julia-baked:0.6
docker tag 292522074875.dkr.ecr.us-east-1.amazonaws.com/julia-baked:0.6 julia-baked:0.6
```

5. Build your desired Dockerimage which contains your experiment:

```
# Build HelloBatch Docker image
export AWS_PROFILE='GPForecasting@Invenia'
docker build -t gpforecastingexperiment:latest .
$(aws ecr get-login --no-include-email --registry-ids 052722095006 --region us-east-1)
REPO=$(grep ecr_uri resources.yml | cut -d' ' -f2)
docker tag gpforecastingexperiment:latest ${REPO}:latest
docker push ${REPO}:latest
```

6. Submit your Batch job with desired experiment script as a string instead of
    "scripts/experiment_template.jl" after the "-s" flag in the command specification.

```
job_definition=$(grep job_definition resources.yml | cut -d' ' -f2)
manager_queue=$(grep manager_queue_arn resources.yml | cut -d' ' -f2)

aws batch submit-job --job-definition $job_definition --job-queue $manager_queue --job-name gpforecasting_job --container-overrides command="["julia", "scripts/experiment.jl", "gpforecasting_job", "-c", "batch", "-n", "2", "-a", "s3", "-s", "scripts/experiment_template.jl"]"
```

7. View results on CloudWatch Logs (click on your job in AWS Batch and click on `logs`).

8. Results saved to the `s3://invenia-gpforecasting` S3 bucket.

## Lengthier explanation under the hood

### Explaining the "resource.yml" file

You can use the "resource.yml" file that points to the pre-made GPForecasting stack. In order to do so,
create a "resource.yml" file inside the root directory of your `GPForecasting` folder (if it does not already exist)
and paste the following inside of it:

This will set up your environment to run jobs with the "GPForecasting" Job Definition (2 vCPUs and 8096 Mb of Memory). With this setup, results of experiments will be saved directly to the `s3://invenia-gpforecasting` S3 bucket folder. If you would like to create/configure your own "resource.yml" file to point to a different Job Definition or different settings, refer to the [`HelloBatch` README](https://gitlab.invenia.ca/invenia/HelloBatch.jl). Note that this step might already be complete if you have cloned a branch already containing the appropriate "resources.yml" file.

If you would like a specific amount of memory of compute power, there are currently two GPForecasting cloudformation job definitions, each with its own memory and price features. They are as follows:

* `GPForecasting Revision1	vCPUs: 2	Memory(Mb): 8096	Image: 052722095006.dkr.ecr.us-east-1.amazonaws.com/hello-batch/gpforecasting`

 Estimated Price for 2 instances (running with `-nprocs = 2`) running for 24 hours: 4.80 USD

 `resources.yml` file required to access this JobDefinition:
```julia
ecr_arn: arn:aws:ecr:us-east-1:052722095006:repository/hello-batch/gpforecasting
ecr_uri: 052722095006.dkr.ecr.us-east-1.amazonaws.com/hello-batch/gpforecasting
job_definition: arn:aws:batch:us-east-1:052722095006:job-definition/GPForecasting:1
batch_output_role: arn:aws:iam::052722095006:role/gpforecasting-JLBatchJobRole-1VBN4AZ1L10V9
compute_arn: arn:aws:batch:us-east-1:052722095006:compute-environment/gpforecasting
manager_queue_arn: arn:aws:batch:us-east-1:052722095006:job-queue/gpforecasting-Managers
worker_queue_arn: arn:aws:batch:us-east-1:052722095006:job-queue/gpforecasting-Workers
s3_uri: s3://invenia-gpforecasting
aws_profile: GPForecasting@Invenia
```

* `JobDefinition-254b0250fdc6f53 Revision6	vCPUs: 2	Memory(Mb): 16000	Image: 052722095006.dkr.ecr.us-east-1.amazonaws.com/gpforecasting/gpforecasting:latest`

 Estimated Price for 2 instances (running with `-nprocs = 2`) running for 24 hours: 19.20 USD

 `resources.yml` file required to access this JobDefinition:
```julia
ecr_arn: arn:aws:ecr:us-east-1:052722095006:repository/hello-batch/gpforecasting
ecr_uri: 052722095006.dkr.ecr.us-east-1.amazonaws.com/gpforecasting/gpforecasting
job_definition: arn:aws:batch:us-east-1:052722095006:job-definition/JobDefinition-254b0250fdc6f53:10
batch_output_role: arn:aws:iam::052722095006:role/gpforecasting-JLBatchJobRole-1VBN4AZ1L10V9
compute_arn: arn:aws:batch:us-east-1:052722095006:compute-environment/gpforecasting
manager_queue_arn: arn:aws:batch:us-east-1:052722095006:job-queue/gpforecasting-Managers
worker_queue_arn: arn:aws:batch:us-east-1:052722095006:job-queue/gpforecasting-Workers
s3_uri: s3://invenia-gpforecasting
aws_profile: GPForecasting@Invenia
```

In short, the first JobDefinition should be used if a low memory (8000 Mb range) experiment should be run,
whereas the second JobDefinition should be used for a larger memory experiment (16000 Mb range) (default behaviour).

If you would like to profile your data to view how much memory you require, run your docker image locally and initiate the experiment with the desired number of processes (i.e. `julia -p 2 scripts/experiment_script.jl`), and run the following commands on another terminal:

```
# Run cAdvisor
docker run \
--volume=/:/rootfs:ro \
--volume=/var/run:/var/run:rw \
--volume=/sys:/sys:ro \
--volume=/var/lib/docker/:/var/lib/docker:ro \
--publish=8080:8080 \
--detach=true \
--name=cadvisor \
google/cadvisor:latest
```

Then go to http://localhost:8080 to view your memory usage.

### Notes on Experiment Definitions

An experiment should currently be defined inside of the `experiment()` function in `src/experiment.jl`. A sample
experiment function that was used for `DAMEC` experiments is included.

Important notes for experiment definitions:

- If using `pmap`, ensure that all variables referenced *inside* of `pmap` that were initialized *outside* should be initialized using the `@everywhere` macro. Ex:

```julia
@everywhere logger = getlogger()
results = pmap(parameters) do params
    info(logger, "Testing")
    return params
end
```

- `pmap` has it's own error handling procedure, that can be specified using the [`on_error` keyword](https://docs.julialang.org/en/release-0.5/stdlib/parallel/). This should be used instead of `try/catch` statements inside of the `pmap` as these can and have resulted in Batch workers exiting.

```julia
@everywhere logger = getlogger()
results = pmap(parameters, on_error = ex -> string(ex)) do params
    info(logger, "Testing")
    return params
end
```

- Results do not need to be saved inside of the experiment definition. The experiment just needs to `return` the results, and these will be automatically saved to S3 (see the `scripts/experiment.jl` script to view )

- `PyCall` requires an extra "tracing" package in order to run inside of a `pmap`. Therefore, the following is added to the `deps/build.jl` file:
```julia
Conda.add("tblib==1.3.2") # PyCall requires this in order to function inside of pmap
```
