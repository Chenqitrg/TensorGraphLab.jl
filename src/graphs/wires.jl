"""An oriented identity wire with one output and one input."""
struct IdentityWire <: AbstractTensorNode{1,1}
    dual::Bool
end

dualflags(w::IdentityWire) = (w.dual, !w.dual)
base_node(w::IdentityWire) = w
_key(w::IdentityWire) = (:wire, w.dual)
_nodekey(w::IdentityWire) = _key(w)
_nodeexpr(w::IdentityWire) = w.dual ? :(TensorGraphLab.IdentityWire(true)) : :id
_replace(w::IdentityWire, replacements) = w
Base.adjoint(w::IdentityWire) = w
Base.transpose(w::IdentityWire) = IdentityWire(!w.dual)

# id[a; b] uses TensorKit's identity symbol in graph notation.
Occurrence(::typeof(TensorKit.id), out::Tuple, inp::Tuple) = Occurrence(IdentityWire(false), out, inp)

# Suppress degree-two identities on a closed seam to recover the original
# lattice vertices. An isolated identity needs an external space specification.
function _suppress_wires(g::TensorGraph, renames::AbstractDict{IndexLabel,IndexLabel}=Dict{IndexLabel,IndexLabel}())
    any(t -> base_node(t.object) isa IdentityWire, occurrences(g)) || return g
    external = Set(_boundary(g))
    neighbors = Dict{IndexLabel,Vector{IndexLabel}}()
    terms = Occurrence[]
    for t in occurrences(g)
        if base_node(t.object) isa IdentityWire
            a, b = t.indices
            push!(get!(neighbors, a, IndexLabel[]), b)
            push!(get!(neighbors, b, IndexLabel[]), a)
        else
            push!(terms, t)
        end
    end
    for index in keys(neighbors)
        haskey(renames, index) && continue
        chain = IndexLabel[index]
        renames[index] = index
        for a in chain, b in neighbors[a]
            haskey(renames, b) && continue
            push!(chain, b)
            renames[b] = index
        end
        ends = filter(in(external), chain)
        length(ends) <= 1 && any(a -> length(neighbors[a]) == 1, chain) ||
            throw(ArgumentError("an isolated identity wire needs a bound space"))
        name = isempty(ends) ? first(chain) : only(ends)
        foreach(a -> renames[a] = name, chain)
    end
    reduced = map(terms) do t
        _occurrence(t.object, map(i -> get(renames, i, i), t.indices))
    end
    return TensorGraph(Tuple(reduced), boundary(g)...; scalar=isscalar(g))
end
