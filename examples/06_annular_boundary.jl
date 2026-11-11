include(joinpath(@__DIR__, "_setup.jl"))
using TensorGraphLab

let
    B = TensorNode(:B, (4, 4))
    @tube annular=(a,b,c) T[u; d] := B[b a c u; d b a c]

    # Traverse the opened boundary: b a c, then u, then c a b, then d.
    @assert T.annular == (:b, :a, :c)
    @assert isplanar(open_graph(T))
    showgraph(T)
    showgraph(open_graph(T); name=:patch)
end
