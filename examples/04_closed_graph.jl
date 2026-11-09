include(joinpath(@__DIR__, "_setup.jl"))
using TensorKit, TensorGraphLab

let
    a = randn(ComplexF64, ℂ^2 ← ℂ^3)
    A = TensorNode(:A,a)
    @graph z = A[i; j] * A'[j; i]
    @graph Z[] := A[i; j] * A'[j; i]
    bindings = @coloring z A=a
    value = evaluate(z,bindings)
    tensor = evaluate(Z,bindings)
    @assert value isa Number && value ≈ norm(a)^2
    @assert tensor isa AbstractTensorMap && numind(tensor) == 0
    @assert scalar(tensor) ≈ value
    showgraph(z; name=:z)
    showgraph(Z; name=:Z)
    println("Scalar value: ", value)
    println("Example 4 passed: scalar and zero-leg tensor outputs.")
end
