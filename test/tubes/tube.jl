@testset "Boundary of the opened tube" begin
    B = TensorNode(:B, (4, 4))
    # The tensor itself is a planar patch. The tuple does not specify wire order.
    @tube annular=(a,b,c) t[u; d] := B[b a c u; d b a c]
    @test t.annular == (:b, :a, :c)
    @test fieldnames(typeof(t)) == (:annular, :patch)
    @test boundary(t) == ((:u,), (:d,))
    @test numind(t) == 2
    printed = Meta.parse(sprint(showgraph, t))
    restored = Core.eval(@__MODULE__, :(let B = $B; $printed; end))
    @test restored == t
    @test restored.annular == t.annular
    @test isplanar(open_graph(t))
    @tube annular=(c,b,a) reordered[u; d] := B[b a c u; d b a c]
    @test t == reordered
    # All cut versions below consist of just one planar tensor. Only the first
    # one above has the boundary of V ⊗ B ← A ⊗ V in the declared leg order.
    @test_throws ArgumentError (@tube annular=(a,b,c) bad[u; d] := B[b a c u; d a b c])
    @test_throws ArgumentError (@tube annular=(a,b,c) bad[u; d] := B[b u a c; d b a c])
    M = TensorNode(:M, (1, 1))
    @tube annular=(a,) trace[] := M[a; a]
    @test (numout(trace), numin(trace)) == (0, 0)
    @test isplanar(open_graph(trace))
    @tube annular=() trivial[o; i] := M[o; i]
    @test isempty(trivial.annular)
    @test boundary(trivial) == boundary(open_graph(trivial))
    @test boundary(trace) == ((), ())
    @test_throws ArgumentError (@tube annular=(i,) bad[o; i] := M[o; i])
    @test_throws ArgumentError (@tube annular=(a,a) bad[] := M[a; a])
end

@testset "Ordered upper and lower tube boundaries" begin
    S = TensorNode(:S, (5, 5))
    @tube annular=(a,b,c) t[o1 o2; i1 i2] := S[b a c o1 o2; i1 i2 b a c]
    @test t.annular == (:b, :a, :c)
    @test_throws ArgumentError (@tube annular=(a,b,c) x[o2 o1; i1 i2] := S[b a c o1 o2; i1 i2 b a c])
    @test_throws ArgumentError (@tube annular=(a,b,c) x[o1 o2; i2 i1] := S[b a c o1 o2; i1 i2 b a c])
    K = TensorNode(:K, (2, 1))
    @tube annular=(a,) out[(o,);()] := K[a o; a]
    @test (numout(out), numin(out)) == (1, 0)
    @tube annular=(a,) input[();(i,)] := K'[a; i a]
    @test (numout(input), numin(input)) == (0, 1)
    @test_throws ArgumentError (@tube annular=(a,) z = S[a b c d e; a b c d e])
end

@testset "Planarity and annular matching are distinct checks" begin
    B = TensorNode(:B, (3, 3))
    @graph planar[l1 l2 o; i r2 r1] := B[l1 l2 o; i r2 r1]
    @test isplanar(planar)
    @tube annular=(a,b) good[o; i] := planar[a b o; i a b]
    @test isplanar(open_graph(good))
    @test_throws ArgumentError (@tube annular=(a,b) crossed[o; i] := planar[a b o; i b a])
    M = TensorNode(:M, (1, 1))
    @graph child[o; i] := M[o; hidden] * M[hidden; i]
    @tube annular=(a,) chain[] := child[a; a]
    @test_throws ArgumentError (@tube annular=(hidden,) bad[o; i] := child[o; i])
    printed = sprint(showgraph, chain)
    @test occursin("@tube annular=(a,)", printed)
    @test Core.eval(@__MODULE__, :(let M = $M; $(Meta.parse(printed)); end)) == chain
end

@testset "Single identity segment across the diagonal seam" begin
    A, B = TensorNode(:A, (2,2)), TensorNode(:B, (2,2))
    @tube annular=(x, r) t[o1 o2; p1 p2] := A[o2 r; y x] * B[x y; p1 p2] * id[o1; r]
    @graph expected[o1 o2; p1 p2] := A[o2 o1; y x] * B[x y; p1 p2]
    @test TensorGraphLab.closed_graph(t) == expected
    @test TubeGraph(t.annular, open_graph(t)) == t
    @test isplanar(open_graph(t))
    @test graphfromdata(graphdata(open_graph(t))) == open_graph(t)
    printed = Meta.parse(sprint(showgraph, t))
    @test Core.eval(@__MODULE__, :(let A = $A, B = $B; $printed; end)) == t
    @test_throws ArgumentError TubeGraph((:a, :a), open_graph(t))
end
