include(joinpath(@__DIR__, "_setup.jl"))
using TensorKit, TensorGraphLab

let
    a = randn(ComplexF64, ℂ^2 ← ℂ^3)
    b = randn(ComplexF64, ℂ^3 ← ℂ^4)
    A, B = TensorNode(:A,a), TensorNode(:B,b)
    @graph G[left; right] := A[left; bond] * B[bond; right]
    bindings = coloring(G, A=>a, B=>b)
    result = evaluate(G,bindings)
    @assert result ≈ a*b
    showgraph(G)
    println("Result space: ", space(result))
    println("Example 1 passed: graph evaluation agrees with matrix composition.")
end
