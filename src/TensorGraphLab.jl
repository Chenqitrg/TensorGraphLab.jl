module TensorGraphLab

using TensorKit
import TensorOperations

const TO = TensorOperations

export TensorNode, TensorGraph, @graph, showgraph, substitute, isplanar
export TubeGraph, @tube, open_graph
export @coloring, coloring, evaluate

include("graphs/nodes.jl")
include("graphs/graphs.jl")
include("graphs/wires.jl")
include("graphs/identity.jl")
include("graphs/substitution.jl")
include("graphs/persistence.jl")
include("graphs/notation.jl")
include("graphs/display.jl")
include("graphs/planarity.jl")

include("tubes/boundary.jl")
include("tubes/tube.jl")
include("tubes/closure.jl")
include("tubes/notation.jl")

include("evaluation/coloring.jl")
include("evaluation/spaces.jl")
include("evaluation/execution.jl")

end
