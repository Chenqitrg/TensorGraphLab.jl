"""An immutable set of color bindings; tensors are borrowed, not copied."""
struct Coloring{T<:NamedTuple}
    tensors::T
end
Base.getindex(c::Coloring, name::Symbol) = getproperty(c.tensors,name)
Base.getindex(c::Coloring, node::TensorNode) = c[label(node)]

_colors(g) = Dict(label(base_node(t.object)) => base_node(t.object) for t in occurrences(expand(g)))

function _check_binding(node, tensor)
    tensor isa AbstractTensorMap || throw(ArgumentError("color $(label(node)) requires an AbstractTensorMap"))
    (numout(node),numin(node)) == (numout(tensor),numin(tensor)) ||
        throw(ArgumentError("port counts do not match color $(label(node))"))
    # Cartesian spaces erase duality; formal graph arrows remain independent.
    if !(spacetype(tensor) <: CartesianSpace)
        dualflags(node) == dualflags(TensorNode(label(node),tensor)) ||
            throw(ArgumentError("dual directions do not match color $(label(node))"))
    end
end

_bind(n::TensorNode,c) = c[label(n)]
_bind(n::AdjointNode,c) = adjoint(_bind(parent(n),c))
_bind(n::PermutedNode,c) = permute(_bind(parent(n),c),n.indices)
_bind(n::TransposedNode,c) = transpose(_bind(parent(n),c),n.indices)

function _prepared(g,c::Coloring)
    nodes = _colors(g)
    Set(keys(c.tensors)) == Set(keys(nodes)) ||
        throw(ArgumentError("provide exactly one binding for every basic color"))
    foreach(name -> _check_binding(nodes[name],c[name]), keys(nodes))
    flat = expand(g)
    tensors = map(t -> _bind(t.object,c),occurrences(flat))
    length(tensors) > 1 && TensorKit.check_spacetype(tensors...)
    ports = Dict{Int,Tuple{Int,Int}}()
    for (v,term) in enumerate(occurrences(flat)), (p,index) in enumerate(term.indices)
        if haskey(ports,index)
            w,q = ports[index]
            TO.checkcontractible(tensors[w],q,false,tensors[v],p,false,index)
        else
            ports[index] = (v,p)
        end
    end
    return flat,tensors
end

"""
    coloring(graph, A=>tensorA, B=>tensorB, ...)

Bind all basic colors at once, using basic nodes or Symbol labels as keys.
Validate ranks, representable dual directions and connected TensorKit spaces.
Views are checked through TensorKit; no transformed tensor is retained.
"""
function coloring(g::TensorGraph, pairs::Pair...)
    nodes = _colors(g)
    labels = Symbol[]
    values = Any[]
    for (key,tensor) in pairs
        key isa Union{Symbol,TensorNode} || throw(ArgumentError("binding keys must be basic nodes or Symbol labels"))
        name = key isa Symbol ? key : label(key)
        haskey(nodes,name) || throw(ArgumentError("unknown color $name"))
        name in labels && throw(ArgumentError("duplicate binding for color $name"))
        key isa TensorNode && key != nodes[name] && throw(ArgumentError("node signature does not match color $name"))
        push!(labels,name)
        push!(values,tensor)
    end
    result = Coloring(NamedTuple{Tuple(labels)}(Tuple(values)))
    validate(g,result)
    return result
end

"""Recheck all bindings and connected spaces without performing a contraction."""
function validate(g::TensorGraph,c::Coloring)
    _prepared(g,c)
    return nothing
end

"""
    @coloring G A=TA B=TB

Bind literal color names to tensors in the caller's scope. A begin block of
assignments is also supported. Each right-hand side is evaluated once.
"""
macro coloring(g, assignments...)
    items = length(assignments) == 1 && assignments[1] isa Expr && assignments[1].head === :block ?
        assignments[1].args : assignments
    pairs = Any[]
    for item in items
        item isa LineNumberNode && continue
        item isa Expr && item.head === :(=) && item.args[1] isa Symbol ||
            throw(ArgumentError("use @coloring G A=tensorA B=tensorB"))
        push!(pairs,:($(QuoteNode(item.args[1])) => $(esc(item.args[2]))))
    end
    return :($coloring($(esc(g)), $(pairs...)))
end
