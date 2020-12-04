using Documenter, GPForecasting

makedocs(;
    modules=[GPForecasting],
    format=Documenter.HTML(
        prettyurls=get(ENV, "CI", nothing) == "true",
        assets=["assets/invenia.css"],
    ),
    pages=[
        "Home" => "index.md",
        "Kernels" => "kernels.md",
        "Extended Input Space" => "extended_input_space.md",
        "Model Notes" => "model_notes.md",
        "Developer Notes" => "developer_notes.md",
        "API" => "api.md",
    ],
    repo="https://gitlab.invenia.ca/invenia/GPForecasting.jl/blob/{commit}{path}#L{line}",
    sitename="GPForecasting.jl",
    authors="Invenia Technical Computing",
    checkdocs=:exports,
    strict=true,
)
