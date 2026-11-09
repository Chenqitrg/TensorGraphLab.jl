@testset "Complete checked coloring" begin
    a = randn(ComplexF64, ℂ^2 ← ℂ^3)
    b = randn(ComplexF64, ℂ^3 ← ℂ^4)
    A, B = TensorNode(:A,a), TensorNode(:B,b)
    @graph G[l;r] := A[l;i] * B[i;r]
    bindings = coloring(G,A=>a,B=>b)
    @test bindings[:A] === a
    @test bindings[B] === b
    @test validate(G,bindings) === nothing
    @test_throws ArgumentError coloring(G,A=>a)
    @test_throws ArgumentError coloring(G,A=>a,B=>b,:C=>a)
    @test_throws ArgumentError coloring(G,A=>a,:A=>a,B=>b)
    @test_throws ArgumentError coloring(G,A=>ones(2,3),B=>b)
    @test_throws ArgumentError coloring(G,A=>randn(ComplexF64,ℂ^2 ⊗ ℂ^2 ← ℂ^3),B=>b)
    @test_throws ArgumentError coloring(G,TensorNode(:A,1,1,(true,false))=>a,B=>b)
    @test_throws ArgumentError coloring(G,A=>randn(ComplexF64,(ℂ^2)' ← ℂ^3),B=>b)
    @test_throws SpaceMismatch coloring(G,A=>a,B=>randn(ComplexF64,ℂ^5 ← ℂ^4))
    calls = Ref(0)
    c = @coloring G A=(calls[] += 1; a) B=b
    @test c[:A] === a && calls[] == 1
    d = @coloring G begin
        A = a
        B = b
    end
    @test d[:B] === b
    @graph H[l;r] := G[l;i] * G'[i;r]
    @test validate(H,coloring(H,:A=>a,:B=>b)) === nothing
    real_tensor = randn(ℝ^2 ← ℝ^3)
    R = TensorNode(:R,1,1,(true,false))
    @graph real_graph[l;r] := R[l;r]
    @test validate(real_graph,coloring(real_graph,R=>real_tensor)) === nothing
    for ex in (:(@coloring G A=>a), :(@coloring G A[1]=a))
        @test_throws ArgumentError macroexpand(@__MODULE__,ex)
    end
end
