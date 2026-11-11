_tupleexpr(values::Tuple) = Expr(:tuple, (v isa Tuple ? _tupleexpr(v) : v for v in values)...)
_group(values) = length(values) == 1 ? only(values) : Expr(isempty(values) ? :tuple : :row, values...)
function _tensorref(object, outputs, inputs)
    isempty(outputs) && isempty(inputs) && return Expr(:ref,object)
    return Expr(:typed_vcat,object,_group(outputs),_group(inputs))
end

# Julia's Expr printer quotes unusual identifiers automatically. Reject keywords
# that it would print as syntax rather than as a variable reference.
function _binding_name(name::Symbol)
    printed = sprint(Base.show_unquoted,name)
    valid = try
        Meta.parse(printed) === name
    catch
        false
    end
    valid || throw(ArgumentError("$name cannot be printed as a binding name"))
    return name
end

_nodeexpr(n::TensorNode) = _binding_name(label(n))
_nodeexpr(n::AdjointNode) = Expr(TO.prime,_nodeexpr(parent(n)))
_nodeexpr(n::PermutedNode) = Expr(:call,:permute,_nodeexpr(parent(n)),_tupleexpr(n.indices))
_nodeexpr(n::TransposedNode) = Expr(:call,:transpose,_nodeexpr(parent(n)),_tupleexpr(n.indices))

function _fresh_name(prefix, used)
    i = 1
    while Symbol(prefix,i) in used
        i += 1
    end
    name = Symbol(prefix,i)
    push!(used,name)
    return name
end

function _display_parts(g::TensorGraph; name::Symbol=:G, depth::Union{Int,Nothing}=nothing, keep=())
    _binding_name(name)
    isnothing(depth) || depth >= 0 || throw(ArgumentError("display depth must be nonnegative"))
    expanded = depth !== 0 && any(t -> t.object isa TensorGraph,occurrences(g))
    visible, names = expanded ? _expand_with_names(g, depth) : (g, nothing)
    original_boundary = (boundary(g)[1]...,boundary(g)[2]...)
    visible_boundary = (boundary(visible)[1]...,boundary(visible)[2]...)
    indices = Dict{Any,Any}(zip(visible_boundary,original_boundary))
    used_indices = Set{Union{Int,Symbol}}(original_boundary)
    if expanded
        for index in keep
            indices[names[index]] = index
            push!(used_indices, index)
        end
    end
    index_name(index) = expanded ? get!(indices,index) do
        _fresh_name("i",used_indices)
    end : index
    used_objects = Set{Symbol}([name])
    for term in occurrences(expand(g))
        push!(used_objects,label(term.object))
    end
    children = Pair{Symbol,TensorGraph}[]
    child_names = Dict{TensorGraph,Symbol}()
    function object_expr(object)
        object isa AbstractTensorNode && return _nodeexpr(object)
        return get!(child_names,object) do
            child_name = _fresh_name("subgraph",used_objects)
            push!(children,child_name=>object)
            child_name
        end
    end
    terms = map(occurrences(visible)) do term
        labels = map(index_name,term.indices)
        m = numout(term.object)
        _tensorref(object_expr(term.object),labels[1:m],labels[m+1:end])
    end
    rhs = length(terms) == 1 ? only(terms) : Expr(:call,:*,terms...)
    lhs = isscalar(g) ? name : _tensorref(name,boundary(g)...)
    return Expr(isscalar(g) ? :(=) : :(:=),lhs,rhs), children
end

"""
    graph_expr(graph; name=:G, depth=nothing)

Return the assignment Expr for @graph. Full expansion is the default. A finite
depth leaves generated subgraph references, which require separate bindings.
Base colors are printed by their labels, not by the caller's original aliases.
"""
graph_expr(g::TensorGraph;kwargs...) = first(_display_parts(g;kwargs...))

"""
    showgraph([io], graph; name=:G, depth=nothing)

Print a readable @graph expression using Julia's Expr printer. Depth 0 retains
direct subgraphs; depth 1 expands one layer; nothing expands everything. Partial
output includes comments identifying the generated child-graph references.
"""
function showgraph(io::IO,g::TensorGraph;kwargs...)
    expression, children = _display_parts(g;kwargs...)
    for (name,child) in children
        println(io,"# ",name," = ",child)
    end
    print(io,"@graph ")
    Base.show_unquoted(io,expression)
    println(io)
    return nothing
end
showgraph(g::TensorGraph;kwargs...) = showgraph(stdout,g;kwargs...)
