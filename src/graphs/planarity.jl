# TensorKit reads an oriented tensor boundary as outputs followed by reversed
# inputs. Its planar complement test joins two such contours along their shared
# indices. Work on indices only, so no coloring or generated code is needed.
function _planar_contour(accept, g::TensorGraph)
    flat = expand(g)
    rings = Vector{Int}[]
    for term in occurrences(flat)
        m = numout(term.object)
        ring = Int[term.indices[1:m]..., reverse(term.indices[m+1:end])...]
        ring = TensorKit.planar_unique2(ring)
        allunique(ring) || return nothing
        push!(rings, ring)
    end
    labels = _boundary(g)
    return _join_contours(rings) do ring
        accept(map(i -> labels[i], ring))
    end
end

function _join_contours(accept, rings)
    length(rings) == 1 && return accept(only(rings))
    # First join connected operands. A legal connected join preserves the cyclic
    # order of every remaining port. Disconnected components admit different
    # placements; try them until a placement has the requested outer boundary.
    connected = false
    for i in eachindex(rings), j in i+1:length(rings)
        isdisjoint(rings[i], rings[j]) && continue
        connected = true
        choices = TensorKit.possible_planar_complements(rings[i], rings[j])
        isempty(choices) && continue
        left, right = first(choices)
        rest = [rings[k] for k in eachindex(rings) if k != i && k != j]
        push!(rest, vcat(left, right))
        return _join_contours(accept, rest)
    end
    connected && return nothing
    for i in eachindex(rings), j in i+1:length(rings)
        rest = [rings[k] for k in eachindex(rings) if k != i && k != j]
        for (left, right) in TensorKit.possible_planar_complements(rings[i], rings[j])
            result = _join_contours(accept, [rest; [vcat(left, right)]])
            isnothing(result) || return result
        end
    end
    return nothing
end

"""Check planar contractions with the declared cyclic boundary, using TensorKit's index rules."""
function isplanar(g::TensorGraph)
    expected = [g.outputs..., reverse(g.inputs)...]
    result = _planar_contour(g) do ring
        (isempty(ring) || TensorKit.iscyclicpermutation(ring, expected)) ? true : nothing
    end
    return !isnothing(result)
end
