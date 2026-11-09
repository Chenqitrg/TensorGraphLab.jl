import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io=devnull)
if !isfile(joinpath(@__DIR__, "..", "Manifest.toml"))
    Pkg.develop([Pkg.PackageSpec(path=normpath(joinpath(@__DIR__, "..", "..", name)))
        for name in ("TensorNodeLab.jl", "TensorGraphCore.jl", "TensorGraphNotation.jl")]; io=devnull)
end
Pkg.instantiate(; allow_autoprecomp=false, io=devnull)
