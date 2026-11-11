_replace(n::TensorNode,replacements) = get(replacements,label(n),n)
_replace(n::AdjointNode,replacements) = adjoint(_replace(parent(n),replacements))
_replace(n::PermutedNode,replacements) = permute(_replace(parent(n),replacements),n.indices)
_replace(n::TransposedNode,replacements) = transpose(_replace(parent(n),replacements),n.indices)
function _replace(g::TensorGraph,replacements)
    terms = map(g.terms) do term
        _occurrence(_replace(term.object,replacements),term.indices)
    end
    return TensorGraph(terms,g.outputs,g.inputs;scalar=g.scalar)
end

"""
    substitute(graph, basic_node => replacement, ...)

Simultaneously replace base colors, retaining each occurrence's outer views.
Keys must be underlying TensorNodes. Values may be basic nodes or views with the
same input/output counts. Inserted values are not traversed again. The resulting
hierarchy is validated, including formal arrows and same-label signatures.
"""
function substitute(g::TensorGraph,pairs::Pair...)
    isempty(pairs) && return g
    available = _base_nodes(g)
    replacements = Dict{Symbol,AbstractTensorNode}()
    for (key,value) in pairs
        key isa TensorNode || throw(ArgumentError("substitution keys must be basic TensorNodes"))
        name = label(key)
        haskey(available,name) || throw(ArgumentError("unknown color $name"))
        available[name] == key || throw(ArgumentError("key metadata does not match color $name"))
        haskey(replacements,name) && throw(ArgumentError("duplicate replacement for color $name"))
        value isa AbstractTensorNode || throw(ArgumentError("replacement for $name must be a node or view"))
        (numout(value),numin(value)) == (numout(key),numin(key)) ||
            throw(ArgumentError("replacement partition does not match color $name"))
        replacements[name] = value
    end
    return _replace(g,replacements)
end
