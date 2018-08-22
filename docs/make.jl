using Documenter, GPForecasting

makedocs(;
    modules=[GPForecasting],
    format=:html,
    pages=[
        "Home" => "index.md",
        "Kernels" => "Kernels.md",
        "Functions" => "Functions.md",
        "Extended input space" => "EIS.md",
        "Experiments" => "Experiments.md",
    ],
    repo="https://gitlab.invenia.ca/research/GPForecasting.jl/blob/{commit}{path}#L{line}",
    sitename="GPForecasting.jl",
    authors="Invenia Technical Computing",
    assets=["assets/invenia.css"],
)
