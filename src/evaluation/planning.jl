# Choose the cheapest available planar pair if the unrestricted optimum
# violates the cyclic order. TensorKit supplies the allowed contour joins.
function _planar_cost_tree(flat::TensorGraph, small::Tuple{Vararg{Int}})
    rings = map(collect(occurrences(flat))) do term
        m = numout(term.object)
        TensorKit.planar_unique2(Int[term.indices[1:m]..., reverse(term.indices[m+1:end])...])
    end
    trees = Any[eachindex(rings)...]
    weight(indices::AbstractVector) = count(i -> !(i in small), indices)
    while length(trees) > 1
        best = nothing
        best_cost = nothing
        for i in eachindex(rings), j in i+1:length(rings)
            options = TensorKit.possible_planar_complements(rings[i], rings[j])
            isempty(options) && continue
            a, b = first(options)
            remaining = vcat(a, b)
            cost = (isdisjoint(rings[i], rings[j]), weight(union(rings[i], rings[j])), weight(remaining))
            if isnothing(best_cost) || cost < best_cost
                best, best_cost = (i, j, remaining), cost
            end
        end
        isnothing(best) && throw(ArgumentError("no planar contraction is available"))
        i, j, remaining = best
        rings[i], trees[i] = remaining, (trees[i], trees[j])
        deleteat!(rings, j)
        deleteat!(trees, j)
    end
    return only(trees)
end
