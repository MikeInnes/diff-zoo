root = joinpath(@__DIR__, "..")
using Pkg; Pkg.activate(root)

src = joinpath(root, "src")
out = joinpath(root, "notebooks")

using Literate

mkpath(out)

for f in ["Project.toml", "Manifest.toml"]
  cp(joinpath(root, f), joinpath(out, f), force = true)
end

function preprocess(s)
  s = "using Pkg; Pkg.activate(\".\"); Pkg.instantiate()\n#-\n" * s
end

for f in ["utils.jl"]
  cp(joinpath(src, f), joinpath(out, f), force = true)
end

for f in ["intro.jl"]
  Literate.notebook(joinpath(src, "intro.jl"), out,
                    preprocess = preprocess,
                    credit = false)
end
