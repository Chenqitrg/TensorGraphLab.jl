# Apply recorded node views to spaces only, never to tensor coefficients.
function _bindspace(n::IndexedColor, c::Coloring)
    _check_binding(n, c[label(n)])
    return space(_bind(n,c))
end

_bindspace(n::TensorNode, c::Coloring) = space(c[label(n)])
_bindspace(n::AdjointNode, c::Coloring) = adjoint(_bindspace(parent(n), c))
_bindspace(n::PermutedNode, c::Coloring) = permute(_bindspace(parent(n), c), n.indices)
_bindspace(n::TransposedNode, c::Coloring) = transpose(_bindspace(parent(n), c), n.indices)

"""
    space(graph::TensorGraph, coloring::Coloring)

Return the TensorKit map space of the bound graph in its declared boundary order.
Validate bindings and connected spaces using space metadata only. No tensor
contraction, permutation of tensor data, or transfer tensor allocation is performed.
A closed graph has the empty product space as both domain and codomain.
"""
function TensorKit.space(g::TensorGraph, c::Coloring)
    flat, spaces, ports = _prepared_spaces(g, c)
    S = spacetype(first(spaces))
    boundary_space(index) = let (v, p) = ports[index]
        spaces[v][p]
    end
    outputs, inputs = boundary(flat)
    cod = ProductSpace{S}(map(boundary_space, outputs))
    dom = ProductSpace{S}(map(i -> dual(boundary_space(i)), inputs))
    return cod ← dom
end

"""Return the space of the physical boundaries after gluing the annular pairs."""
TensorKit.space(t::TubeGraph, c::Coloring) = space(closed_graph(t), c)
