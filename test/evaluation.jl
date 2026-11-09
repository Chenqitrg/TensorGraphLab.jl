@testset "Evaluation delegates to tensor and planar" begin
    a = randn(ComplexF64,ℂ^2 ← ℂ^3)
    b = randn(ComplexF64,ℂ^3 ← ℂ^4)
    A,B = TensorNode(:A,a),TensorNode(:B,b)
    @graph G[l;r] := A[l;i] * B[i;r]
    c = @coloring G A=a B=b
    @tensor expected[l;r] := a[l;i] * b[i;r]
    @test evaluate(G,c) ≈ expected
    @test evaluate(G,c;method=:planar) ≈ expected
    @test_throws ArgumentError evaluate(G,c;method=:unknown)
    @graph H[l;r] := G[l;i] * G'[i;r]
    @test evaluate(H,c) ≈ expected * expected'
    @test evaluate(H,c;method=:planar) ≈ expected * expected'
    @graph z = A[i;j] * A'[j;i]
    @graph Z[] := A[i;j] * A'[j;i]
    @test evaluate(z,coloring(z,A=>a)) ≈ norm(a)^2
    @test evaluate(z,coloring(z,A=>a);method=:planar) ≈ norm(a)^2
    zero_rank = evaluate(Z,coloring(Z,A=>a))
    @test zero_rank isa AbstractTensorMap && numind(zero_rank) == 0
    @test scalar(zero_rank) ≈ norm(a)^2
    @test scalar(evaluate(Z,coloring(Z,A=>a);method=:planar)) ≈ norm(a)^2
    @graph view[l;r] := transpose(permute(A,((2,),(1,)))')[l;r]
    @test evaluate(view,coloring(view,A=>a)) ≈ transpose(permute(a,((2,),(1,)))')
    @test evaluate(view,coloring(view,A=>a);method=:planar) ≈ transpose(permute(a,((2,),(1,)))')
    # Cached code must use live data, including after mutation and new bindings.
    for (_,block) in blocks(a)
        block .*= 2
    end
    @test evaluate(G,c) ≈ 2expected
    a2 = randn(ComplexF64,ℂ^5 ← ℂ^3)
    @test evaluate(G,coloring(G,A=>a2,B=>b)) ≈ a2*b
    clear_evaluation_cache!()
    @test evaluate(G,c) ≈ a*b
    # Labels are data, even when they have a special meaning in upstream macros.
    for name in (:τ, Symbol("a b"), :end)
        N = TensorNode(name,a)
        @graph named[l;r] := N[l;r]
        @test evaluate(named,coloring(named,N=>a);method=:planar) ≈ a
    end
    # Reusing bindings with another graph cannot bypass space validation.
    @graph incompatible[l;r] := A[l;i] * B'[i;r]
    @test_throws SpaceMismatch evaluate(incompatible,c)
    r = randn(ℝ^2 ← ℝ^3)
    R = TensorNode(:R,r)
    @graph real_view[l;r] := transpose(R)[l;r]
    @test evaluate(real_view,coloring(real_view,R=>r)) ≈ transpose(r)
    # Compile a previously unseen graph inside an already running function.
    function local_evaluation(a)
        N = TensorNode(:fresh,a)
        @graph fresh[l;r] := N[l;r]
        evaluate(fresh,coloring(fresh,N=>a))
    end
    @test local_evaluation(a) ≈ a
end

@testset "Symmetry and planar ordering" begin
    V = Z2Space(0=>2,1=>1)
    a = randn(ComplexF64,V ← V)
    b = randn(ComplexF64,V ← V)
    A,B = TensorNode(:A,a),TensorNode(:B,b)
    @graph G[l;r] := A[l;i] * B[i;r]
    @planar expected[l;r] := a[l;i] * b[i;r]
    @test evaluate(G,coloring(G,A=>a,B=>b);method=:planar) ≈ expected
    @test evaluate(G,coloring(G,A=>a,B=>b)) ≈ a*b
    q = randn(ComplexF64,ℂ^2 ⊗ ℂ^3 ← ℂ^4 ⊗ ℂ^5)
    Q = TensorNode(:Q,q)
    @graph crossing[b a;c d] := Q[a b;c d]
    @tensor crossed[b a;c d] := q[a b;c d]
    @test evaluate(crossing,coloring(crossing,Q=>q)) ≈ crossed
    @test_throws ArgumentError evaluate(crossing,coloring(crossing,Q=>q);method=:planar)
    # Canonically equal graphs can have different planar factor orders.
    @graph forward[l;r] := A[l;i] * B[i;r]
    @graph reversed[l;r] := B[i;r] * A[l;i]
    @test forward == reversed
    @planar reversed_expected[l;r] := b[i;r] * a[l;i]
    @test evaluate(forward,coloring(forward,A=>a,B=>b);method=:planar) ≈ expected
    @test evaluate(reversed,coloring(reversed,A=>a,B=>b);method=:planar) ≈ reversed_expected
end

@testset "Boundary ranks and braided planar tensors" begin
    for (m,n) in ((0,0),(0,1),(1,0),(0,2),(2,0))
        codom = ProductSpace{ComplexSpace}(ntuple(_ -> ℂ^2,m))
        dom = ProductSpace{ComplexSpace}(ntuple(_ -> ℂ^2,n))
        tensor = randn(ComplexF64,codom ← dom)
        node = TensorNode(:N,tensor)
        # Programmatic construction covers ranks that differ in each fixture.
        outputs,inputs = Tuple(1:m),Tuple(m+1:m+n)
        graph = TensorGraph((Occurrence(node,outputs,inputs),),outputs,inputs)
        c = coloring(graph,node=>tensor)
        @test evaluate(graph,c) ≈ tensor
        @test evaluate(graph,c;method=:planar) ≈ tensor
    end
    V = Vect[FibonacciAnyon](:I=>1,:τ=>1)
    a = randn(ComplexF64,V ⊗ V ← V ⊗ V)
    b = randn(ComplexF64,V ⊗ V ← V ⊗ V)
    A,B = TensorNode(:A,a),TensorNode(:B,b)
    @graph G[l1 l2;r1 r2] := A[l1 l2;i j] * B[i j;r1 r2]
    @planar expected[l1 l2;r1 r2] := a[l1 l2;i j] * b[i j;r1 r2]
    @test evaluate(G,coloring(G,A=>a,B=>b);method=:planar) ≈ expected
    braid = BraidingTensor(V ⊗ V ← V ⊗ V)
    τ = TensorNode(:τ,braid)
    @graph crossing[l1 l2;r1 r2] := τ[l1 l2;r1 r2]
    @test evaluate(crossing,coloring(crossing,τ=>braid);method=:planar) ≈ braid
end
