_nodekey(n::TensorNode) = (:node, label(n), numout(n), numin(n), dualflags(n))
_nodekey(n::AdjointNode) = (:adjoint, _nodekey(parent(n)))
_nodekey(n::PermutedNode) = (:permute, _nodekey(parent(n)), n.indices)
_nodekey(n::TransposedNode) = (:transpose, _nodekey(parent(n)), n.indices)

function _links(g)
    ports = _portmap(g)
    external = Dict(zip(_boundary(g), 1:numind(g)))
    return map(enumerate(g.terms)) do (v,term)
        map(enumerate(term.indices)) do (p,index)
            if haskey(external,index)
                (0,external[index])
            else
                ends = ports[index]
                ends[1] == (v,p) ? ends[2] : ends[1]
            end
        end
    end
end

# Ordered ports fix traversal once the starting vertex has been chosen.
function _walk(links, root)
    order = [root]
    numbering = Dict(root => 1)
    position = 1
    while position <= length(order)
        for (neighbor, _) in links[order[position]]
            if neighbor != 0 && !haskey(numbering,neighbor)
                push!(order,neighbor)
                numbering[neighbor] = length(order)
            end
        end
        position += 1
    end
    return order, numbering
end

function _rooted_key(g, links, root)
    order, numbering = _walk(links,root)
    records = Tuple((_nodekey(g.terms[v].object),
        Tuple(w == 0 ? (0,p) : (numbering[w],p) for (w,p) in links[v])) for v in order)
    return repr(records)
end

"""
    canonical_key(graph)

Return an immutable structural key, ignoring dummy names and hierarchical factoring.
Each connected component is canonically rooted using ordered ports; disconnected
components are sorted. This includes closed networks and repeated colors. The key
is for this schema's structural comparisons, not a cross-version file identifier.
"""
function canonical_key(graph::TensorGraph)
    g = expand(graph)
    links = _links(g)
    seen = falses(length(g.terms))
    components = String[]
    for root in eachindex(g.terms)
        seen[root] && continue
        vertices, _ = _walk(links,root)
        seen[vertices] .= true
        push!(components, minimum(_rooted_key(g,links,v) for v in vertices))
    end
    return (numout(g), numin(g), g.scalar, Tuple(sort!(components)))
end

Base.:(==)(a::TensorGraph,b::TensorGraph) = canonical_key(a) == canonical_key(b)
Base.isequal(a::TensorGraph,b::TensorGraph) = isequal(canonical_key(a),canonical_key(b))
Base.hash(g::TensorGraph,seed::UInt) = hash(canonical_key(g),seed)
