@testset "Readable graph expressions" begin
    A = TensorNode(:A,1,1,(false,true))
    B = TensorNode(:B,1,1,(false,true))
    @graph G[left; right] := A[left; bond] * B[bond; right]
    text = sprint(showgraph,G)
    @test occursin("@graph G",text)
    @test occursin("A[left; bond]",text)
    @test occursin("B[bond; right]",text)
    @test TensorGraphLab.TO.getlhs(graph_expr(G)) == :(G[left;right])
    @graph H[left;right] := G[left;link] * G[link;right]
    full = sprint(io -> showgraph(io,H;name=:H))
    @test count("A[",full) == 2
    @test count("B[",full) == 2
    @test !occursin("subgraph",full)
    partial = sprint(io -> showgraph(io,H;name=:H,depth=0))
    @test occursin("subgraph1",partial)
    @test !occursin("subgraph2",partial) # Repeated child values share a display binding.
    @test_throws ArgumentError graph_expr(H;depth=-1)
    @test_throws ArgumentError graph_expr(H;name=:end)
    # Rebuild full expressions in an ordinary lexical scope with the base bindings.
    build(ex,a,b) = Base.invokelatest(Core.eval(@__MODULE__, :((A,B) -> begin
        $(Expr(:macrocall,Symbol("@graph"),LineNumberNode(0),ex))
    end)),a,b)
    @test build(graph_expr(H),A,B) == H
    @graph Z[] := A[i;i]
    @graph z = A[i;i]
    @test !isscalar(build(graph_expr(Z),A,B))
    @test isscalar(build(graph_expr(z),A,B))
    @graph views[l;r] := transpose(permute(A,((2,),(1,)))')[l;r]
    @test build(graph_expr(views),A,B) == views
    @graph collision[i1;i2] := G[i1;j] * G[j;i2]
    @test build(graph_expr(collision),A,B) == collision
    @graph numeric[-1;-2] := G[-1;j] * G[j;-2]
    @test build(graph_expr(numeric),A,B) == numeric
    C = TensorNode(Symbol("a b"),1,1,(false,true))
    @graph awkward[l;r] := C[l;r]
    @test occursin("var\"a b\"",sprint(showgraph,awkward))
    @test Meta.parse(sprint(showgraph,awkward)) isa Expr
    @graph ranks[-1 -2; -3 -4] := TensorNode(:Q,2,2,(false,false,true,true))[-1 -2; -3 -4]
    @test TensorGraphLab.TO.istensor(TensorGraphLab.TO.getlhs(graph_expr(ranks)))
end
