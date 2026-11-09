const IndexLabel = Union{Int,Symbol}
_check_labels(labels) = all(x -> x isa IndexLabel, labels) ||
    throw(ArgumentError("index labels must be Int or Symbol"))

"""One occurrence of a node or child graph, with indices ordered as output then input."""
struct Occurrence{T,L<:Tuple}
    object::T
    indices::L
    function Occurrence(object, outputs::Tuple, inputs::Tuple)
        object isa Union{AbstractTensorNode,TensorGraph} ||
            throw(ArgumentError("an occurrence requires a node or a graph"))
        (length(outputs), length(inputs)) == (numout(object), numin(object)) ||
            throw(ArgumentError("occurrence partition does not match its object"))
        indices = (outputs..., inputs...)
        _check_labels(indices)
        return new{typeof(object),typeof(indices)}(object, indices)
    end
end

"""
    TensorGraph(terms::Tuple, outputs::Tuple, inputs::Tuple; scalar=false)

An immutable hierarchy of oriented occurrences. Internal indices occur twice,
with opposite formal arrows; boundary indices occur once. Index names are local
to each graph. `scalar=true` distinguishes scalar intent from a zero-leg tensor.
"""
struct TensorGraph{M,N,T<:Tuple,O<:Tuple,I<:Tuple}
    terms::T
    outputs::O
    inputs::I
    scalar::Bool
    function TensorGraph(terms::Tuple, outputs::Tuple, inputs::Tuple; scalar::Bool=false)
        !isempty(terms) && all(t -> t isa Occurrence, terms) ||
            throw(ArgumentError("a graph requires a nonempty tuple of occurrences"))
        _check_labels((outputs..., inputs...))
        scalar && !(isempty(outputs) && isempty(inputs)) &&
            throw(ArgumentError("scalar output requires an empty boundary"))
        g = new{length(outputs),length(inputs),typeof(terms),typeof(outputs),typeof(inputs)}(
            terms, outputs, inputs, scalar)
        _validate(g)
        return g
    end
end

occurrences(g::TensorGraph) = g.terms
boundary(g::TensorGraph) = (g.outputs, g.inputs)
isscalar(g::TensorGraph) = g.scalar
TensorKit.numout(::TensorGraph{M,N}) where {M,N} = M
TensorKit.numin(::TensorGraph{M,N}) where {M,N} = N
TensorKit.numind(::TensorGraph{M,N}) where {M,N} = M + N
_boundary(g) = (g.outputs..., g.inputs...)

function _portmap(g)
    ports = Dict{IndexLabel,Vector{Tuple{Int,Int}}}()
    for (v, term) in enumerate(g.terms), (p, index) in enumerate(term.indices)
        push!(get!(ports, index, Tuple{Int,Int}[]), (v,p))
    end
    return ports
end

function dualflags(g::TensorGraph)
    ports = _portmap(g)
    return map(_boundary(g)) do index
        v, p = only(ports[index])
        dualflags(g.terms[v].object)[p]
    end
end

function _validate(g)
    external = _boundary(g)
    length(unique(external)) == length(external) ||
        throw(ArgumentError("boundary indices must be unique"))
    ports = _portmap(g)
    all(i -> haskey(ports,i), external) || throw(ArgumentError("unused boundary index"))
    for (index, ends) in ports
        if index in external
            length(ends) == 1 || throw(ArgumentError("boundary index $index must occur once"))
        else
            length(ends) == 2 || throw(ArgumentError("internal index $index must occur twice"))
            (v,p), (w,q) = ends
            dualflags(g.terms[v].object)[p] != dualflags(g.terms[w].object)[q] ||
                throw(ArgumentError("internal index $index joins equal formal arrows"))
        end
    end
    _base_nodes(g) # Also check colors across subgraph boundaries.
    return g
end

function _base_nodes(g)
    nodes = Dict{Symbol,TensorNode}()
    function visit(object)
        if object isa TensorGraph
            foreach(t -> visit(t.object), object.terms)
        else
            node = base_node(object)
            name = label(node)
            haskey(nodes,name) && nodes[name] != node &&
                throw(ArgumentError("conflicting base-node signatures for color $name"))
            nodes[name] = node
        end
    end
    visit(g)
    return nodes
end

function _occurrence(object, indices)
    outputs = ntuple(i -> indices[i], numout(object))
    inputs = ntuple(i -> indices[numout(object)+i], numin(object))
    return Occurrence(object, outputs, inputs)
end

function _flatten!(terms, g, names, nextindex, depth)
    rename(index) = get!(names, index) do
        nextindex[] += 1
    end
    for term in g.terms
        indices = map(rename, term.indices)
        if term.object isa TensorGraph && depth !== 0
            childnames = Dict{IndexLabel,Int}(zip(_boundary(term.object), indices))
            _flatten!(terms, term.object, childnames, nextindex, isnothing(depth) ? nothing : depth-1)
        else
            push!(terms, _occurrence(term.object, indices))
        end
    end
    return terms
end

"""Expand child graphs to `depth` layers (default: all), renaming boundary and private indices hygienically."""
function expand(g::TensorGraph; depth::Union{Int,Nothing}=nothing)
    isnothing(depth) || depth >= 0 || throw(ArgumentError("expansion depth must be nonnegative"))
    names = Dict{IndexLabel,Int}(zip(_boundary(g), 1:numind(g)))
    terms = _flatten!(Occurrence[], g, names, Ref(numind(g)), depth)
    return TensorGraph(Tuple(terms), Tuple(1:numout(g)), Tuple(numout(g)+1:numind(g)); scalar=g.scalar)
end

function Base.adjoint(g::TensorGraph)
    terms = map(reverse(g.terms)) do term
        nout, nin = numout(term.object), numin(term.object)
        Occurrence(adjoint(term.object), ntuple(i -> term.indices[nout+i], nin),
            ntuple(i -> term.indices[i], nout))
    end
    return TensorGraph(terms, g.inputs, g.outputs; scalar=g.scalar)
end

function Base.show(io::IO, g::TensorGraph{M,N}) where {M,N}
    print(io, "TensorGraph{", M, ",", N, "}(", length(g.terms), " occurrences, ",
        g.scalar ? "scalar" : "tensor", " output)")
end
