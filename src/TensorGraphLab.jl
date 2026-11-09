module TensorGraphLab

using TensorKit
import TensorOperations

const TO = TensorOperations

# Everyday graph construction, binding and evaluation.
export TensorNode, TensorGraph
export @graph, @coloring, coloring, evaluate
export showgraph, substitute

include("nodes.jl")
include("graphs.jl")
include("identity.jl")
include("substitution.jl")
include("persistence.jl")
include("notation.jl")
include("display.jl")
include("coloring.jl")
include("execution.jl")

end
