include(joinpath(@__DIR__, "_setup.jl"))
using TensorKit, TensorGraphLab

let
    V = Vect[FibonacciAnyon](:I=>1,:τ=>1)
    a = randn(ComplexF64,V ⊗ V ← V ⊗ V)
    b = randn(ComplexF64,V ⊗ V ← V ⊗ V)
    A, B = TensorNode(:A,a), TensorNode(:B,b)
    @graph G[l1 l2; r1 r2] := A[l1 l2; i j] * B[i j; r1 r2]
    bindings = @coloring G A=a B=b
    result = evaluate(G,bindings; method=:planar)
    @planar expected[l1 l2; r1 r2] := a[l1 l2; i j] * b[i j; r1 r2]
    @assert result ≈ expected
    showgraph(G)
    println("Result sectors: ", collect(blocksectors(result)))
    println("Example 5 passed: planar evaluation with Fibonacci symmetry.")
end
