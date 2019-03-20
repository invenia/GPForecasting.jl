using Documenter, GPForecasting

makedocs(;
    modules=[GPForecasting],
    format=Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    pages=[
        "Home" => "index.md",
        "Kernels" => "Kernels.md",
        "Functions" => "Functions.md",
        "Extended Input Space" => "EIS.md",
        "Nodes and Trees" => "Nodes.md",
    ],
    repo="https://gitlab.invenia.ca/research/GPForecasting.jl/blob/{commit}{path}#L{line}",
    sitename="GPForecasting.jl",
    authors="Invenia Technical Computing",
    assets=["assets/invenia.css"],
    checkdocs=:exports,
)
