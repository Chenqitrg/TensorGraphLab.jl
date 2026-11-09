using Test, TensorKit, TensorGraphLab
for file in (isempty(ARGS) ? ("coloring", "evaluation") : ARGS)
    include(file * ".jl")
end
