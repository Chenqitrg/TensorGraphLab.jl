import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io=devnull)
Pkg.instantiate(; allow_autoprecomp=false, io=devnull)
