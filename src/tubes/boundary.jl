# Split only the two occurrences visible in this scope. Child graphs keep their
# private labels, even when those labels have the same spelling.
function _open_annular_edges(g, annular)
    _check_labels(annular)
    allunique(annular) || throw(ArgumentError("duplicate annular index"))
    ports = _portmap(g)
    names = Dict{Tuple{Int,Int},Symbol}()
    used = Set{IndexLabel}(keys(ports))
    original = Dict{Symbol,IndexLabel}()
    for a in annular
        ends = get(ports, a, ())
        length(ends) == 2 || throw(ArgumentError(
            "annular index $a must occur twice in the current graph scope"))
        for port in ends
            endpoint = _fresh_name(string("cut_", a, "_"), used)
            names[port] = endpoint
            original[endpoint] = a
        end
    end
    terms = ntuple(length(g.terms)) do v
        term = g.terms[v]
        indices = ntuple(p -> get(names, (v,p), term.indices[p]), length(term.indices))
        _occurrence(term.object, indices)
    end
    # A temporary boundary exposes all endpoints while searching for their
    # physical order; its order is not used as a planarity constraint.
    opened = TensorGraph(terms, (keys(original)..., g.outputs...), g.inputs)
    return opened, original
end

function _tube_boundary(ring, g, k, original)
    if isempty(ring)
        return ((), (), ())
    end
    m, n = numout(g), numin(g)
    for start in eachindex(ring)
        at(i) = ring[mod1(start + i - 1, length(ring))]
        all(i -> at(k+i) == g.outputs[i], 1:m) || continue
        all(i -> at(2k+m+i) == g.inputs[n-i+1], 1:n) || continue
        left = ntuple(at, k)
        # Read the opposite side backwards so that its endpoints match the
        # first side in tensor-product order.
        right = ntuple(i -> at(2k+m+1-i), k)
        all(x -> haskey(original, x), left) || continue
        order = map(x -> original[x], left)
        allunique(order) || continue
        all(i -> get(original, right[i], nothing) == order[i], 1:k) || continue
        return left, right, order
    end
    return nothing
end
