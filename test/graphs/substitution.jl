@testset "Simultaneous basic-color substitution" begin
    A = TensorNode(:A, 1, 1, (false, true))
    B = TensorNode(:B, 1, 1, (false, true))
    G = TensorGraph((Occurrence(A,(:l,),(:i,)), Occurrence(B,(:i,),(:r,))),(:l,),(:r,))
    swapped = TensorGraph((Occurrence(B,(:l,),(:i,)), Occurrence(A,(:i,),(:r,))),(:l,),(:r,))
    @test substitute(G, A=>B, B=>A) == swapped
    @test substitute(G) === G
    @test occurrences(G)[1].object === A
    H = TensorGraph((Occurrence(G,(:a,),(:b,)), Occurrence(G,(:b,),(:c,))),(:a,),(:c,))
    @test substitute(H,A=>B,B=>A) == TensorGraph((Occurrence(swapped,(:a,),(:b,)),Occurrence(swapped,(:b,),(:c,))),(:a,),(:c,))
    p = ((2,), (1,))
    views = TensorGraph((Occurrence(A,(:a,),(:b,)), Occurrence(A',(:c,),(:d,)),
        Occurrence(transpose(A),(:e,),(:f,)), Occurrence(permute(A,p),(:g,),(:h,))), (:a,:c,:e,:g), (:b,:d,:f,:h))
    result = substitute(views, A=>B')
    objects = map(o -> o.object, occurrences(result))
    @test objects == (B', B, transpose(B'), permute(B',p))
    @test map(o -> o.object, occurrences(substitute(views, A=>permute(A,p)))) ==
        (permute(A,p), permute(A,p)', transpose(permute(A,p)), permute(permute(A,p),p))
    @test_throws ArgumentError substitute(G,A'=>B)
    @test_throws ArgumentError substitute(G,:A=>B)
    @test_throws ArgumentError substitute(G,A=>B,A=>A)
    @test_throws ArgumentError substitute(G,TensorNode(:C,1,1,(false,true))=>B)
    @test_throws ArgumentError substitute(G,TensorNode(:A,1,1,(true,false))=>B)
    @test_throws ArgumentError substitute(G,A=>TensorNode(:C,2,0,(false,true)))
    @test_throws ArgumentError substitute(G,A=>G)
    @test_throws ArgumentError substitute(G,A=>transpose(B))
end
