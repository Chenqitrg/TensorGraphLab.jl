module TensorGraphLab
using TensorKit, TensorGraphNotation
import TensorOperations
const TO = TensorOperations
for name in names(TensorGraphNotation)
    name === :TensorGraphNotation && continue
    @eval export $name
end
export Coloring, coloring, @coloring, validate, evaluate, clear_evaluation_cache!
include("coloring.jl")
include("execution.jl")
end
