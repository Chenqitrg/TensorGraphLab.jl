include(joinpath(@__DIR__, "_setup.jl"))
using TensorKit, TensorGraphLab

let
    a = randn(ComplexF64, ℂ^2 ← ℂ^2)
    b = randn(ComplexF64, ℂ^2 ← ℂ^2)
    A, B = TensorNode(:A,a), TensorNode(:B,b)
    @graph G[left; right] := A'[left; bond] * B[bond; right]
    swapped = substitute(G,A=>B,B=>A)
    bindings = @coloring G A=a B=b
    @assert evaluate(G,bindings) ≈ a'*b
    @assert evaluate(swapped,bindings) ≈ b'*a
    @graph transposed[left; right] := transpose(A)[left; right]
    @assert evaluate(transposed,coloring(transposed,A=>a)) ≈ transpose(a)
    showgraph(swapped; name=:swapped)
    println("Example 3 passed: node views and simultaneous color replacement.")
end
