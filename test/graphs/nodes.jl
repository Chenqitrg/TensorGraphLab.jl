
# Independent reference: inspect numerical TensorKit tensors, not node internals.
flags(t) = Tuple(isdual(space(t, i)) for i in 1:numind(t))
has_payload(x) = x isa Union{AbstractTensorMap, AbstractArray, TensorKit.VectorSpace} ||
    any(has_payload(getfield(x, i)) for i in 1:fieldcount(typeof(x)))
function check_interface(node, t)
    @test numout(node) == numout(t)
    @test numin(node) == numin(t)
    @test dualflags(node) == flags(t)
end

@testset "Tensor nodes" begin
    @testset "Default directions from an output/input rank pair" begin
        A = TensorNode(:A, (2, 1))
        @test A isa TensorNode{2,1}
        @test label(A) == :A
        @test dualflags(A) == (false, false, true)
        @test dualflags(A') == (false, true, true)
        @test dualflags(transpose(A)) == (true, false, false)
        @test !has_payload(A)
        @test TensorNode(:Z, (0, 0)) isa TensorNode{0,0}
        @test dualflags(TensorNode(:ket, (2, 0))) == (false, false)
        @test dualflags(TensorNode(:bra, (0, 2))) == (true, true)
        @test_throws ArgumentError TensorNode(:A, (-1, 2))
        @test_throws ArgumentError TensorNode(:A, (2, -1))
    end

    @testset "Typed views, immediate parents and exact identities" begin
        A = TensorNode(:A, 2, 1, (false, true, true))
        P = permute(A, ((3,), (1, 2)))
        T = transpose(A)
        @test A'' === A
        @test P'' === P
        @test T'' === T
        @test parent(A') === A
        @test parent(P) === A
        @test parent(T) === A
        @test parent(P') === P
        @test base_node(P') === A
        @test A' isa AdjointNode{1,2}
        @test P isa PermutedNode{1,2}
        @test T isa TransposedNode{1,2}
        @test_throws TypeError AdjointNode{2,1,typeof(A)}
        for node in (A, P, T, A')
            identity_partition = (Tuple(1:numout(node)), Tuple(numout(node)+1:numind(node)))
            @test permute(node, identity_partition) === node
            @test transpose(node, identity_partition) === node
        end
        Z = TensorNode(:Z, 0, 0, ())
        @test Z'' === Z
        @test permute(Z, ((), ())) === Z
        @test transpose(Z) === Z
        @test length(Set([A, A'', permute(A, ((1, 2), (3,)))])) == 1
    end

    @testset "Input and output ranks are separate type parameters" begin
        A = TensorNode(:A, 2, 1, (false, false, true))
        B = TensorNode(:B, 3, 0, (false, false, true))
        @test typeof(A) != typeof(B) # Equal total rank must not erase the partition.
        @test A isa TensorNode{2,1}
        @test A isa AbstractTensorNode{2,1}
        @test @inferred(adjoint(A)) isa AbstractTensorNode{1,2}
        @test @inferred(transpose(A)) isa AbstractTensorNode{1,2}
        @test @inferred(permute(A, ((3,), (1, 2)))) isa AbstractTensorNode{1,2}
        @test TensorNode(:Z, 0, 0, ()) isa AbstractTensorNode{0,0}
        t = ones(ComplexF64, ℂ^2 ⊗ ℂ^2 ← ℂ^2)
        @test @inferred(TensorNode(:A, t)) isa TensorNode{2,1}
    end

    @testset "Metadata without dimensions or tensor ownership" begin
        V = ℂ^2
        t = ones(ComplexF64, V ⊗ V' ← V' ⊗ V)
        A = TensorNode(:A, t)
        check_interface(A, t)
        @test dualflags(A) == (false, true, false, true)
        @test label(A) == :A
        @test A == TensorNode(:A, 2, 2, (false, true, false, true))
        W = ℂ^3
        @test A == TensorNode(:A, ones(ComplexF64, W ⊗ W' ← W' ⊗ W))
        @test A != TensorNode(:B, t)
        @test length(Set([A, TensorNode(:A, t)])) == 1
        @test isimmutable(A)
        @test !has_payload(A)
        @test_throws ArgumentError TensorNode(:A, -1, 2, (false,))
        @test_throws ArgumentError TensorNode(:A, 2, 2, (false,))
        @test_throws MethodError TensorNode(:A, zeros(2, 2))
    end

    @testset "Direction operations agree with TensorKit" begin
        for V in (ℂ^2, Rep[ℤ₂](0=>1, 1=>1))
            t = ones(ComplexF64, V ⊗ V' ← V)
            A = TensorNode(:A, t)
            p = ((3, 1), (2,))
            q = ((3,), (2, 1))
            check_interface(A', t')
            check_interface(permute(A, p), permute(t, p))
            check_interface(transpose(A), transpose(t))
            check_interface(transpose(A, q), transpose(t, q))
            check_interface(transpose(permute(A, p)'), transpose(permute(t, p)'))
            @test base_node(transpose(A')) === A
            @test label(permute(A, p)) == :A
            @test A' != transpose(A) # different operations, even with equal interfaces
            @test A'' === A
            @test numind(A'') == numind(A)
            @test dualflags(A'') == dualflags(A)
            @test length(Set([transpose(A), transpose(TensorNode(:A, t))])) == 1
            @test isimmutable(transpose(A'))
            @test !has_payload(transpose(A'))
        end
        A = TensorNode(:A, 2, 2, (false, false, true, true))
        @test_throws ArgumentError permute(A, ((1, 1), (3, 4)))
        @test_throws ArgumentError permute(A, ((1,), (2,)))
        @test_throws ArgumentError transpose(A, ((2, 1), (3, 4)))
        @test_throws ArgumentError transpose(A, ((1,), (2,)))
        @test_throws ArgumentError permute(A, ((0, 2), (3, 4)))
        @test_throws ArgumentError permute(A, ((1, 2), (3, 5)))
    end

    @testset "Real numerical spaces do not erase graph arrows" begin
        t = ones(Float64, ℝ^2 ⊗ ℝ^3 ← ℝ^4)
        A = TensorNode(:A, t)
        @test flags(t) == (false, false, false)
        @test dualflags(A) == (false, false, true)
        @test dualflags(A') == (false, true, true)
        @test (numout(A'), numin(A')) == (1, 2)
        p = ((3, 1), (2,))
        @test dualflags(permute(A, p)) == (true, false, false)
        @test dualflags(transpose(A)) == (true, false, false)
        @test base_node(permute(A, p)) === A
        @test A == TensorNode(:A, 2, 1, (false, false, true))
        # A numerical real tensor has no history from which to recover these arrows.
        @test dualflags(TensorNode(:A, permute(t, p))) == (false, false, true)
        @test dualflags(permute(A, p)) != dualflags(TensorNode(:A, permute(t, p)))
    end

    @testset "Zero legs and readable operation records" begin
        Z = TensorNode(:Z, 0, 0, ())
        @test numind(Z') == 0
        @test dualflags(transpose(Z)) == ()
        @test numind(permute(Z, ((), ()))) == 0
        A = TensorNode(:A, 1, 1, (false, true))
        @test base_node(A) === A
        @test occursin("A", sprint(show, A))
        @test occursin("permute", sprint(show, permute(A, ((2,), (1,)))))
        @test occursin("transpose", sprint(show, transpose(A)))
        @test occursin("adjoint", sprint(show, A'))
    end
end
