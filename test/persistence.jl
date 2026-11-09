@testset "Versioned structural persistence" begin
    A = TensorNode(:τ,1,1,(false,true))
    G = TensorGraph((Occurrence(A',(:l,),(:i,)),Occurrence(A,(:i,),(:r,))),(:l,),(:r,))
    H = TensorGraph((Occurrence(G,(:a,),(:b,)),Occurrence(G',(:b,),(:c,))),(:a,),(:c,))
    data = graphdata(H)
    @test data.version == 1
    restored = graphfromdata(data)
    @test restored == H
    @test occurrences(restored)[1].object isa TensorGraph
    @test boundary(restored) == boundary(H)
    io = IOBuffer()
    serialize(io, data)
    seekstart(io)
    @test graphfromdata(deserialize(io)) == H
    Z = TensorGraph((Occurrence(A,(:i,),(:i,)),),(),();scalar=true)
    @test isscalar(graphfromdata(graphdata(Z)))
    @test_throws ArgumentError graphfromdata(merge(data, (version=999,)))
    @test_throws ArgumentError graphfromdata((version=1,))
    @test_throws ArgumentError graphfromdata("run arbitrary code")
    @test_throws ArgumentError graphfromdata(merge(data, (outputs=(:missing,),)))
    @test_throws ArgumentError graphfromdata(merge(data, (version=true,)))
    saved = graphdata(G)
    node = (:node, :A, 1, 1, (false, true))
    corrupt_node(definition) = merge(saved, (terms=(
        (object=(kind=:node,definition=definition),indices=(:l,:r)),),))
    @test_throws ArgumentError graphfromdata(corrupt_node((:node,:A,-1,3,(false,true))))
    @test_throws ArgumentError graphfromdata(corrupt_node((:node,:A,1,1,(false,))))
    @test_throws ArgumentError graphfromdata(corrupt_node((:node,:A,1,1,(0,1))))
    @test_throws ArgumentError graphfromdata(corrupt_node((:permute,node,((1,),(1,)))))
    @test_throws ArgumentError graphfromdata(corrupt_node((:adjoint,node,node)))
    broken_occurrence = merge(saved,(terms=((object=saved.terms[1].object,),),))
    @test_throws ArgumentError graphfromdata(broken_occurrence)
    wrong_rank = merge(saved,(terms=(merge(saved.terms[1],(indices=(:l,),)),),))
    @test_throws ArgumentError graphfromdata(wrong_rank)
    broken_child = merge(data.terms[1].object,(outputs=(:missing,),))
    broken_nested = merge(data,(terms=(merge(data.terms[1],(object=broken_child,)),data.terms[2]),))
    @test_throws ArgumentError graphfromdata(broken_nested)
    S = TensorNode(:S,0,0,())
    scalar_graph = TensorGraph((Occurrence(S,(),()),),(),();scalar=true)
    @test graphfromdata(graphdata(scalar_graph')) == scalar_graph'
    @test graphfromdata(graphdata(scalar_graph'')) == scalar_graph
    R = TensorGraph((Occurrence(transpose(A),(:l,),(:r,)),),(:l,),(:r,))
    P = TensorGraph((Occurrence(permute(A,((2,),(1,))),(:l,),(:r,)),),(:l,),(:r,))
    @test graphfromdata(graphdata(R)) == R
    @test graphfromdata(graphdata(P)) == P
    @test occursin("TensorGraph", sprint(show,H))
end
