# Install the correct branch of HelloBatch
info("$(Pkg.installed("HelloBatch"))")
Pkg.checkout("HelloBatch", "ls/getrevinfo") # Checkout the branch with working function saving for GPs
info("$(Pkg.installed("HelloBatch"))")
