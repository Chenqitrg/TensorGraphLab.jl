include(joinpath(@__DIR__, "_setup.jl"))
using TensorKit, TensorGraphLab

let
    a = randn(ComplexF64, ℂ^2 ⊗ ℂ^2 ← ℂ^2 ⊗ ℂ^2)
    A = TensorNode(:A,a)
    @graph row[l1 l2; r1 r2] := A[l1 l2; i j] * A[i j; r1 r2]
    @graph transfer[l1 l2; r1 r2] := row[l1 l2; i j] * row[i j; r1 r2]
    bindings = @coloring transfer A=a
    result = evaluate(transfer,bindings)
    @assert result ≈ a*a*a*a
    showgraph(transfer; name=:transfer)
    # Reusing cached code must still read the current tensor entries.
    for (_,block) in blocks(a)
        block .*= 2
    end
    @assert evaluate(transfer,bindings) ≈ 16result
    println("Example 2 passed: nested transfer and live tensor data.")
end
