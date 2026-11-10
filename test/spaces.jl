# This tensor has no numerical storage: static queries must only inspect its space.
struct MetadataTensor{S,M,N} <: AbstractTensorMap{Float64,S,M,N}
    tensor_space::TensorMapSpace{S,M,N}
end
TensorKit.space(t::MetadataTensor) = t.tensor_space

@testset "Bound graph spaces without tensor data" begin
    a = MetadataTensor(ℂ^2 ⊗ (ℂ^3)' ← ℂ^4 ⊗ (ℂ^5)')
    A = TensorNode(:A, a)
    @graph G[w s; n e] := A[w s; n e]
    c = coloring(G, A => a)
    @test space(G, c) == space(a)
    @test space(G', c) == adjoint(space(a))

    p = ((3, 1), (4, 2))
    P = permute(A, p)
    @graph permuted[w s; n e] := P[w s; n e]
    @test space(permuted, coloring(permuted, A => a)) == permute(space(a), p)

    Q = transpose(A)
    @graph transposed[w s; n e] := Q[w s; n e]
    @test space(transposed, coloring(transposed, A => a)) == transpose(space(a))
    @graph nested[w s; n e] := G[w s; i j] * G'[i j; n e]
    @test space(nested, coloring(nested, A => a)) == (codomain(a) ← codomain(a))

    bad = Coloring((A = MetadataTensor(ℂ^2 ⊗ (ℂ^3)' ← ℂ^4 ⊗ ℂ^5),))
    @test_throws ArgumentError space(G, bad)
end
