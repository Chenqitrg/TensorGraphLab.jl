@testset "Canonical colored ordered-port structure" begin
    A = TensorNode(:A, 1, 1, (false, true))
    B = TensorNode(:B, 1, 1, (false, true))
    G = TensorGraph((Occurrence(A, (:l,), (:x,)), Occurrence(B, (:x,), (:r,))), (:l,), (:r,))
    renamed = TensorGraph((Occurrence(B, (10,), (30,)), Occurrence(A, (20,), (10,))), (20,), (30,))
    @test G == renamed
    @test isequal(G, renamed) && hash(G) == hash(renamed)
    H = TensorGraph((Occurrence(G, (:a,), (:b,)), Occurrence(G, (:b,), (:c,))), (:a,), (:c,))
    @test H == expand(H)
    @test H'' == H
    @test length(Set([H, expand(H), H''])) == 1
    @test G != TensorGraph(reverse(occurrences(G)), (:r,), (:l,))
    @test G != TensorGraph((Occurrence(B, (:l,), (:x,)), Occurrence(A, (:x,), (:r,))), (:l,), (:r,))
    cycle = (Occurrence(A, (:i,), (:j,)), Occurrence(A, (:j,), (:k,)), Occurrence(A, (:k,), (:i,)))
    C = TensorGraph(cycle, (), ())
    for order in ((1,2,3), (1,3,2), (2,1,3), (2,3,1), (3,1,2), (3,2,1))
        other = TensorGraph(map(i -> cycle[i], order), (), ())
        @test canonical_key(C) == canonical_key(other)
    end
    loops = TensorGraph((Occurrence(A, (:i,), (:i,)), Occurrence(A, (:j,), (:j,)), Occurrence(A, (:k,), (:k,))), (), ())
    @test C != loops
    @test loops == TensorGraph(reverse(occurrences(loops)), (), ())
    @test C != TensorGraph(cycle, (), (); scalar=true)
    Q = TensorNode(:Q, 2, 2, (false, false, true, true))
    X = TensorGraph((Occurrence(Q, (:l,:m), (:i,:j)), Occurrence(Q, (:i,:j), (:r,:s))), (:l,:m), (:r,:s))
    Y = TensorGraph((Occurrence(Q, (:l,:m), (:i,:j)), Occurrence(Q, (:j,:i), (:r,:s))), (:l,:m), (:r,:s))
    @test X != Y
    @test X != TensorGraph(occurrences(X), (:m,:l), (:r,:s))
    S = TensorNode(:S, 0, 0, ())
    @test TensorGraph((Occurrence(S,(),()), Occurrence(A,(:i,),(:i,))),(),()) ==
          TensorGraph((Occurrence(A,(:k,),(:k,)), Occurrence(S,(),())),(),())
end
