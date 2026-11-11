const TensorFamily = Tuple{Vararg{AbstractTensorMap}}
const TensorBinding = Union{AbstractTensorMap,TensorFamily}
const TensorBindings = NamedTuple{names,<:Tuple{Vararg{TensorBinding}}} where {names}

"""
    Coloring(bindings)

Each named color binds a TensorKit map or an ordered tuple of maps for indexed
occurrences. All maps have the same sector type, available as `sectortype(c)`
or `sectortype(typeof(c))`. An empty coloring has sector type `Union{}`.
"""
struct Coloring{I<:Sector,T<:TensorBindings}
    tensors::T
    function Coloring(bindings::T) where {T<:TensorBindings}
        I = foldl(_binding_sector, values(bindings); init=Union{})
        return new{I,T}(bindings)
    end
end

TensorKit.sectortype(::Type{<:Coloring{I}}) where {I} = I

function _binding_sector(I::Type, tensor::AbstractTensorMap)
    J = sectortype(tensor)
    I === Union{} || I === J || throw(SpaceMismatch("color bindings mix sector types $I and $J"))
    return J
end
_binding_sector(I::Type, family::TensorFamily) = foldl(_binding_sector, family; init=I)

Base.getindex(c::Coloring, name::Symbol) = getproperty(c.tensors,name)
Base.getindex(c::Coloring, node::TensorNode) = c[label(node)]

_colors(g::TensorGraph) = _base_nodes(g)

_colors(t::TubeGraph) = _colors(open_graph(t))

_check_binding(node::TensorNode, value) =
    throw(ArgumentError("color $(label(node)) requires an AbstractTensorMap"))

function _check_binding(node::TensorNode, tensor::AbstractTensorMap)
    (numout(node),numin(node)) == (numout(tensor),numin(tensor)) ||
        throw(ArgumentError("port counts do not match color $(label(node))"))
    # Cartesian spaces erase duality; formal graph arrows remain independent.
    if !(spacetype(tensor) <: CartesianSpace)
        dualflags(node) == dualflags(TensorNode(label(node),tensor)) ||
            throw(ArgumentError("dual directions do not match color $(label(node))"))
    end
end

_check_binding(n::IndexedColor, value) = throw(ArgumentError("indexed color $(label(n)) requires a tensor family"))
function _check_binding(n::IndexedColor, family::TensorFamily)
    _check_binding(n.node, family[n.slot])
end

_bind(n::IndexedColor, c::Coloring) = c[label(n)][n.slot]
_bind(n::TensorNode, c::Coloring) = c[label(n)]
_bind(n::AdjointNode, c::Coloring) = adjoint(_bind(parent(n),c))
_bind(n::PermutedNode, c::Coloring) = permute(_bind(parent(n),c),n.indices)
_bind(n::TransposedNode, c::Coloring) = transpose(_bind(parent(n),c),n.indices)

function _prepared_spaces(g::TensorGraph, c::Coloring)
    return _prepared_spaces(g,c,_suppress_wires(expand(g)))
end

function _prepared_spaces(g::TensorGraph, c::Coloring, flat::TensorGraph)
    nodes = _colors(g)
    Set(keys(c.tensors)) == Set(keys(nodes)) ||
        throw(ArgumentError("provide exactly one binding for every basic color"))
    foreach(name -> _check_binding(nodes[name],c[name]), keys(nodes))
    spaces = map(t -> _bindspace(t.object,c),occurrences(flat))
    length(spaces) > 1 && TensorKit.check_spacetype(spaces...)
    ports = Dict{Int,Tuple{Int,Int}}()
    for (v,term) in enumerate(occurrences(flat)), (p,index) in enumerate(term.indices)
        if haskey(ports,index)
            w,q = ports[index]
            left, right = dual(spaces[w][q]), spaces[v][p]
            left == right || throw(SpaceMismatch("incompatible spaces for $index: $left ≠ $right"))
        else
            ports[index] = (v,p)
        end
    end
    return flat,spaces,ports
end

"""
    coloring(graph, A=>tensorA, B=>tensorB, ...)

Bind all basic colors at once, using basic nodes or Symbol labels as keys.
Validate ranks, representable dual directions and connected TensorKit spaces.
Tube bindings also check each pair of spaces identified by annular closure.
Views are checked through TensorKit spaces without transforming tensor data.
"""
function coloring(g::Union{TensorGraph,TubeGraph}, pairs::Pair...)
    nodes = _colors(g)
    labels = Symbol[]
    values = Any[]
    for (key,tensor) in pairs
        key isa Union{Symbol,TensorNode} || throw(ArgumentError("binding keys must be basic nodes or Symbol labels"))
        name = key isa Symbol ? key : label(key)
        haskey(nodes,name) || throw(ArgumentError("unknown color $name"))
        name in labels && throw(ArgumentError("duplicate binding for color $name"))
        key isa TensorNode && key != nodes[name] && throw(ArgumentError("node signature does not match color $name"))
        node = nodes[name]
        binding = node isa IndexedColor && tensor isa AbstractVector ? Tuple(tensor) : tensor
        binding isa TensorBinding || throw(ArgumentError("color $name requires a TensorMap or a tuple of TensorMaps"))
        push!(labels,name)
        push!(values,binding)
    end
    result = Coloring(NamedTuple{Tuple(labels)}(Tuple(values)))
    validate(g,result)
    return result
end

"""Recheck all bindings and connected spaces without performing a contraction."""
function validate(g::TensorGraph,c::Coloring)
    _prepared_spaces(g,c)
    return nothing
end

function validate(t::TubeGraph, c::Coloring)
    validate(closed_graph(t), c)
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
