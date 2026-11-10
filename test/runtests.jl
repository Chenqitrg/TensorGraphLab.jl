using Test, TensorKit, TensorGraphLab, Serialization

# Explicitly import advanced interfaces exercised by the full test suite.
using TensorGraphLab: AbstractTensorNode, AdjointNode, PermutedNode, TransposedNode
using TensorGraphLab: label, dualflags, base_node, Occurrence, occurrences, boundary
using TensorGraphLab: isscalar, expand, canonical_key, graphdata, graphfromdata, graph_expr
using TensorGraphLab: Coloring, validate, clear_evaluation_cache!

suites = ("nodes", "graphs", "identity", "substitution", "persistence",
          "notation", "display", "coloring", "spaces", "evaluation")
for file in (isempty(ARGS) ? suites : ARGS)
    include(file * ".jl")
end
