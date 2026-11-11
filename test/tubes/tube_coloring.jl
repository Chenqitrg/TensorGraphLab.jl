@testset "Coloring an annular tube" begin
    V, W, U, D = ℂ^2, ℂ^3, ℂ^4, ℂ^5
    a = MetadataTensor(V ⊗ W ⊗ U ← D ⊗ V ⊗ W)
    A = TensorNode(:A, a)
    @tube annular=(y,x) t[u; d] := A[x y u; d x y]
    bindings = coloring(t, A => a)
    @test bindings[:A] === a
    @test (@coloring t A=a)[A] === a
    @test validate(t, bindings) === nothing
    @test space(t, bindings) == (U ← D)
    @test_throws ArgumentError coloring(t)
    @test_throws ArgumentError coloring(t, :A => a, :A => a)

    # Swapping the two annular input spaces preserves the patch ranks.
    swapped = MetadataTensor(V ⊗ W ⊗ U ← D ⊗ W ⊗ V)
    patch_bindings = coloring(open_graph(t), A => swapped)
    @test_throws SpaceMismatch validate(t, patch_bindings)
    @test_throws SpaceMismatch space(t, patch_bindings)
    @test_throws SpaceMismatch coloring(t, A => swapped)

    @graph child[o1 o2 o3; i1 i2 i3] := A[o1 o2 o3; i1 i2 i3]
    @tube annular=(x,y) nested[u; d] := child[x y u; d x y]
    @test validate(nested, bindings) === nothing
    @tube annular=() uncut[o1 o2 o3; i1 i2 i3] := child[o1 o2 o3; i1 i2 i3]
    @test validate(uncut, patch_bindings) === nothing
end
