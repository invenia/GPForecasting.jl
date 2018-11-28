hello_batch_installed = try
    Pkg.installed("HelloBatch") !== nothing
catch
    false
end

if !hello_batch_installed
    Pkg.clone("HelloBatch")
    Pkg.checkout("HelloBatch", "lm/gpf")  # Expect commit: 4c57df1
    Pkg.build("HelloBatch")
else
    info("HelloBatch is already installed")
    Pkg.status("HelloBatch")
end
