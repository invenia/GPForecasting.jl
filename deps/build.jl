# Install the correct branch of HelloBatch
try
    Pkg.installed("HelloBatch")
catch
    Pkg.clone("git@gitlab.invenia.ca:invenia/HelloBatch.jl.git")
    Pkg.checkout("HelloBatch", "ls/getrevinfo") # Checkout the branch with working function saving for GPs
    Pkg.build("HelloBatch")
end

#using Conda

# We have tested up to scipy 1.0.0
#Conda.add("scipy==1.0.0")
#Conda.add("tblib==1.3.2") # PyCall requires this in order to function inside of pmap

#Conda.update()

#info("Verifying scipy install.")
#python_bin = joinpath(Conda.PYTHONDIR, "python")
#run(`$python_bin -c "import scipy.optimize"`)

# ENV["PYTHON"] = Conda.PYTHONDIR
# Fixed issue by adding ENV["PYTHON"] = "" in Dockerfile
# Same for CI files
#Pkg.build("PyCall")
#
Pkg.checkout("Nabla")
# Pkg.checkout("Impute")
Pkg.checkout("ModelAnalysis")
Pkg.build("ModelAnalysis")
