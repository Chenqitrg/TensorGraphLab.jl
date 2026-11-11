@testset "TensorOperations-compatible graph notation" begin
    A = TensorNode(:A,1,1,(false,true))
    B = TensorNode(:B,1,1,(false,true))
    @graph G[left; right] := A[left; bond] * B[bond; right]
    expected = TensorGraph((Occurrence(A,(:left,),(:bond,)),Occurrence(B,(:bond,),(:right,))),(:left,),(:right,))
    @test G == expected
    @graph H[left; right] := G[left; bond] * G[bond; right]
    @test length(occurrences(expand(H))) == 4
    @graph flat[l;r] := A[l;a] * B[a;b] * A[b;c] * B[c;r]
    @test H == flat
    @graph Z[] := A[i;i]
    @graph z = A[i;i]
    @test !isscalar(Z) && isscalar(z) && Z != z
    @graph prime[i';j] := A[i';j]
    @test boundary(prime) == ((Symbol("i′"),),(:j,))
    @graph neg[-1;-2] ≔ A[-1;-2]
    @test boundary(neg) == ((-1,),(-2,))
    @graph tuples[(l,);(r,)] := A[(l,);(r,)]
    @test tuples == neg
    Q = TensorNode(:Q,2,2,(false,false,true,true))
    @graph transfer[-1 -2; -3 -4] := Q[-1 -2; i j] * Q[i j; -3 -4]
    @test transfer isa TensorGraph{2,2}
    @graph view[l;r] := A'[l;r]
    @test occurrences(view)[1].object == A'
    @graph rearranged[l;r] := transpose(A)[l;r]
    @test dualflags(rearranged) == (true,false)
    @graph nested_adjoint[l;r] := G'[l;r]
    @test nested_adjoint == G'
    τ = TensorNode(:τ,1,1,(false,true))
    @graph braidname[l;r] := τ[l;r]
    @test label(occurrences(braidname)[1].object) == :τ
    calls = Ref(0)
    make_node() = (calls[] += 1; A)
    @graph once[l;r] := make_node()[l;i] * make_node()[i;r]
    @test calls[] == 1
    @test length(occurrences(once)) == 2
    X = A
    @graph alias[l;r] := X[l;r]
    @test label(occurrences(alias)[1].object) == :A
    @graph G[l;r] := G[l;i] * B[i;r]
    @test length(occurrences(expand(G))) == 3
    @test length(occurrences(expand(H))) == 4
    local_graph = let A = TensorNode(:local,1,1,(false,true)), l=10, r=20
        @graph R[l;r] := A[l;r]
        R
    end
    @test boundary(local_graph) == ((:l,),(:r,))
    @test label(occurrences(local_graph)[1].object) == :local
    @graph begin
        first[l;r] := A[l;r]
        second[l;r] := first[l;i] * B[i;r]
    end
    @test length(occurrences(expand(second))) == 2
    for ex in (:(@graph bad[l;r] := A[l;r] + B[l;r]),
               :(@graph bad[l;r] := 2 * A[l;r]),
               :(@graph bad[l;r] += A[l;r]),
               :(@graph bad[l;r] = A[l;r]))
        @test_throws ArgumentError macroexpand(@__MODULE__,ex)
    end
    @test_throws ArgumentError (@graph bad[l;r] := A[l;r] * B[l;r])
    @test_throws ArgumentError (@graph bad[l;r] := A[l r])
    @test_throws UndefVarError (@graph bad[l;r] := definitely_undefined_node[l;r])
    @test_throws ArgumentError (@graph bad[l;r] := ones(2,2)[l;r])
end

@testset "TensorKit expression compatibility" begin
    a = randn(ComplexF64, ℂ^2 ← ℂ^3)
    b = randn(ComplexF64, ℂ^3 ← ℂ^4)
    A, B = TensorNode(:A,a), TensorNode(:B,b)
    @graph G[left;right] := A[left;bond] * B[bond;right]
    @tensor c[left;right] := a[left;bond] * b[bond;right]
    @test (numout(G),numin(G),dualflags(G)) == (numout(c),numin(c),dualflags(TensorNode(:C,c)))
    @graph H[left;right] := A'[left;bond] * A[bond;right]
    @tensor d[left;right] := a'[left;bond] * a[bond;right]
    @test dualflags(H) == dualflags(TensorNode(:D,d))
end
