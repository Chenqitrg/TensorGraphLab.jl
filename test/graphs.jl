@testset "Graph construction and hygienic expansion" begin
    A = TensorNode(:A, 1, 1, (false, true))
    B = TensorNode(:B, 1, 1, (false, true))
    G = TensorGraph((Occurrence(A, (:l,), (:x,)), Occurrence(B, (:x,), (:r,))), (:l,), (:r,))
    @test G isa TensorGraph{1,1}
    @test (numout(G), numin(G), numind(G)) == (1, 1, 2)
    @test dualflags(G) == (false, true)
    @test boundary(G) == ((:l,), (:r,))
    @test !isscalar(G)
    @test isimmutable(G)
    @test occurrences(G)[1].object === A
    H = TensorGraph((Occurrence(G, (:l,), (:x,)), Occurrence(G, (:x,), (:r,))), (:l,), (:r,))
    @test occurrences(H)[1].object === G
    F = expand(H)
    @test length(occurrences(expand(H;depth=0))) == 2
    @test occurrences(expand(H;depth=0))[1].object === G
    @test all(t.object isa TensorNode for t in occurrences(expand(H;depth=1)))
    J = TensorGraph((Occurrence(H,(:a,),(:b,)),Occurrence(H,(:b,),(:c,))),(:a,),(:c,))
    @test length(occurrences(expand(J;depth=1))) == 4
    @test all(t.object isa TensorGraph for t in occurrences(expand(J;depth=1)))
    @test all(t.object isa TensorNode for t in occurrences(expand(J;depth=2)))
    @test_throws ArgumentError expand(H;depth=-1)
    @test length(occurrences(F)) == 4
    @test boundary(F) == ((1,), (2,))
    counts = Dict{Int,Int}()
    for term in occurrences(F), index in term.indices
        counts[index] = get(counts, index, 0) + 1
    end
    @test sort(collect(values(counts))) == [1, 1, 2, 2, 2]
    @test all(term.object isa TensorNode for term in occurrences(F))
    @test dualflags(G') == (false, true)
    @test base_node(occurrences(G')[1].object) == B
    @test_throws ArgumentError Occurrence(A, (:i, :j), ())
    @test_throws ArgumentError Occurrence(A, ([1],), (:j,))
    @test_throws ArgumentError TensorGraph((), (), ())
    @test_throws ArgumentError TensorGraph((Occurrence(A, (:i,), (:j,)),), (:i,), ())
    @test_throws ArgumentError TensorGraph((Occurrence(A, (:i,), (:j,)),), (:i,), (:i,))
    @test_throws ArgumentError TensorGraph((Occurrence(A, (:i,), (:j,)),), (:i,), (:z,))
    @test_throws ArgumentError TensorGraph((Occurrence(A, (:i,), (:j,)),), (:i,), (:j,); scalar=true)
    bad = TensorNode(:B, 1, 1, (true, true))
    @test_throws ArgumentError TensorGraph((Occurrence(A, (:l,), (:x,)), Occurrence(bad, (:x,), (:r,))), (:l,), (:r,))
    conflict = TensorNode(:A, 1, 1, (true, false))
    @test_throws ArgumentError TensorGraph((Occurrence(A, (:l,), (:r,)), Occurrence(conflict, (:s,), (:t,))), (:l, :s), (:r, :t))
    Z = TensorGraph((Occurrence(A, (:i,), (:i,)),), (), ())
    z = TensorGraph(occurrences(Z), (), (); scalar=true)
    @test dualflags(Z) == ()
    @test !isscalar(Z) && isscalar(z) && isscalar(z')
    S = TensorNode(:S, 0, 0, ())
    @test length(occurrences(expand(TensorGraph((Occurrence(S, (), ()),), (), ())))) == 1
end
