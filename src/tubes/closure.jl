# Restore the marked contractions in this scope for @tube notation.
function _closed_graph(t::TubeGraph)
    patch = t.patch
    isempty(t.annular) && return patch
    labels = Dict{IndexLabel,IndexLabel}()
    for (i, name) in enumerate(t.annular)
        labels[patch.outputs[i]] = name
        labels[patch.inputs[numin(t) + i]] = name
    end
    terms = map(occurrences(patch)) do term
        _occurrence(term.object, map(index -> get(labels, index, index), term.indices))
    end
    return TensorGraph(terms, boundary(t)...)
end

"""Recover the physical tensor network by gluing the annular legs and suppressing identity wires."""
function closed_graph(t::TubeGraph)
    g = _closed_graph(t)
    all(t -> t.object isa AbstractTensorNode, occurrences(g)) && return _suppress_wires(g)
    any(term -> base_node(term.object) isa IdentityWire, occurrences(expand(g))) || return g
    # Preserve external names when expanding nested occurrences.
    flat, names = _expand_with_names(g, nothing)
    external = Dict{Int,IndexLabel}(names[i] => i for i in _boundary(g))
    used = Set{IndexLabel}(_boundary(g))
    function rename(i)
        return get!(external, i) do
            _fresh_name("bond", used)
        end
    end
    terms = map(t -> _occurrence(t.object, map(rename, t.indices)), flat.terms)
    return _suppress_wires(TensorGraph(terms, boundary(g)...))
end
