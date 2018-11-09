# Install the correct branch of HelloBatch
try
    Pkg.installed("HelloBatch")
catch
    Pkg.clone("git@gitlab.invenia.ca:invenia/HelloBatch.jl.git")
    Pkg.checkout("HelloBatch", "ls/getrevinfo") # Checkout the branch with working function saving for GPs
    Pkg.build("HelloBatch")
end
