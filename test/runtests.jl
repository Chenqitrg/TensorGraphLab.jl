using Test, TensorKit, TensorGraphLab, Serialization

# Explicitly import advanced interfaces exercised by the full test suite.
using TensorGraphLab: AbstractTensorNode, AdjointNode, PermutedNode, TransposedNode
using TensorGraphLab: label, dualflags, base_node, Occurrence, occurrences, boundary
using TensorGraphLab: isscalar, expand, canonical_key, graphdata, graphfromdata, graph_expr
using TensorGraphLab: Coloring, validate, clear_evaluation_cache!

include("support/metadata_tensor.jl")

suites = (
    graphs = ("nodes", "graphs", "identity", "substitution", "persistence", "notation", "display", "planarity"),
    tubes = ("tube", "tube_coloring"),
    evaluation = ("coloring", "spaces", "evaluation"),
)
# Select a directory with a trailing slash or a single suite by its short name.
selectors = String[]
for (group, files) in pairs(suites)
    folder = string(group, "/")
    push!(selectors, folder, files...)
end
for arg in ARGS
    arg in selectors || throw(ArgumentError("unknown test selection: $arg"))
end
for (group, files) in pairs(suites), file in files
    if isempty(ARGS) || string(group, "/") in ARGS || file in ARGS
        include(joinpath(string(group), file * ".jl"))
    end
end
