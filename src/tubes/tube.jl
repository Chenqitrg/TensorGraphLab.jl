"""
    TubeGraph(graph; annular=())
    @tube annular=(a,b) T[outputs; inputs] := ...

Open the marked contractions and check the boundary of a planar representative
of the tube morphism. In cyclic order the boundary must read L, outputs,
reverse(R), reverse(inputs), where L and R contain matching endpoints in the
same order. The patch determines the order of the stored annular labels.
For `k = length(t.annular)`, glue the first `k` outputs of `t.patch` to its
last `k` inputs in matching order to recover the tube.
"""
struct TubeGraph{A<:Tuple,P<:TensorGraph}
    annular::A
    patch::P
    function TubeGraph(g::TensorGraph; annular::Tuple=())
        isscalar(g) && throw(ArgumentError("a tube requires an indexed tensor boundary"))
        opened, original = _open_annular_edges(g, annular)
        halves = _planar_contour(opened) do ring
            _tube_boundary(ring, g, length(annular), original)
        end
        isnothing(halves) && throw(ArgumentError(
            "no planar tube boundary has matching opposite annular sides separating the ordered outputs and inputs"))
        left, right, order = halves
        patch = TensorGraph(opened.terms, (left..., g.outputs...), (g.inputs..., right...))
        return new{typeof(order),typeof(patch)}(order, patch)
    end
end

"""Return the validated planar representative with its annular legs open."""
open_graph(t::TubeGraph) = t.patch
TensorKit.numout(t::TubeGraph) = numout(t.patch) - length(t.annular)
TensorKit.numin(t::TubeGraph) = numin(t.patch) - length(t.annular)
TensorKit.numind(t::TubeGraph) = numout(t) + numin(t)
function boundary(t::TubeGraph)
    outputs, inputs = boundary(t.patch)
    k = length(t.annular)
    return outputs[k+1:end], inputs[1:end-k]
end
_tube_key(t::TubeGraph) = (numout(t), numin(t), t.patch)
Base.:(==)(a::TubeGraph, b::TubeGraph) = _tube_key(a) == _tube_key(b)
Base.isequal(a::TubeGraph, b::TubeGraph) = a == b
Base.hash(t::TubeGraph, h::UInt) = hash(_tube_key(t), h)
