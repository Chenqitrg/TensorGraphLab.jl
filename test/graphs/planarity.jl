@testset "Planarity of disconnected graphs" begin
    M = TensorNode(:M, (1, 1))
    @graph outer[a b c; x y z] := M[a; x] * M[b; y] * M[c; z]
    @test isplanar(outer)
    @graph crossing[a b c; x y z] := M[a; x] * M[b; z] * M[c; y]
    @test !isplanar(crossing)
end
