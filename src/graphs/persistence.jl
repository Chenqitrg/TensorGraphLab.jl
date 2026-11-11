_encode(n::AbstractTensorNode) = (kind=:node, definition=_nodekey(n))
_encode(g::TensorGraph) = graphdata(g)

"""Export versioned plain structural data, retaining hierarchy but no tensors or caches."""
function graphdata(g::TensorGraph)
    terms = map(t -> (object=_encode(t.object),indices=t.indices),g.terms)
    return (kind=:graph,version=1,terms=terms,outputs=g.outputs,inputs=g.inputs,scalar=g.scalar)
end

function _require_fields(data, fields)
    data isa NamedTuple && Set(keys(data)) == Set(fields) ||
        throw(ArgumentError("invalid structural record; expected fields $fields"))
end

function _read_node(definition)
    definition isa Tuple && !isempty(definition) || throw(ArgumentError("invalid node definition"))
    kind = first(definition)
    if kind === :wire && length(definition) == 2 && definition[2] isa Bool
        return IdentityWire(definition[2])
    elseif kind === :node
        length(definition) == 5 || throw(ArgumentError("invalid basic-node record"))
        _, name, nout, nin, duals = definition
        name isa Symbol && nout isa Int && nin isa Int && duals isa Tuple{Vararg{Bool}} ||
            throw(ArgumentError("invalid basic-node metadata"))
        return TensorNode(name,nout,nin,duals)
    elseif kind === :indexed && length(definition) == 3
        return IndexedColor(_read_node(definition[2]), definition[3])
    elseif kind === :adjoint
        length(definition) == 2 || throw(ArgumentError("invalid adjoint record"))
        return adjoint(_read_node(definition[2]))
    elseif kind in (:permute,:transpose)
        length(definition) == 3 && definition[3] isa TensorKit.Index2Tuple ||
            throw(ArgumentError("invalid port transformation record"))
        node, indices = _read_node(definition[2]), definition[3]
        return kind === :permute ? permute(node,indices) : transpose(node,indices)
    end
    throw(ArgumentError("unknown node kind $kind"))
end

function _read_term(data)
    _require_fields(data,(:object,:indices))
    object = _decode(data.object)
    data.indices isa Tuple && length(data.indices) == numind(object) ||
        throw(ArgumentError("invalid occurrence indices"))
    return _occurrence(object,data.indices)
end

function _decode(data)
    data isa NamedTuple && hasproperty(data,:kind) || throw(ArgumentError("missing structural kind"))
    if data.kind === :node
        _require_fields(data,(:kind,:definition))
        return _read_node(data.definition)
    elseif data.kind === :graph
        _require_fields(data,(:kind,:version,:terms,:outputs,:inputs,:scalar))
        data.version === 1 || throw(ArgumentError("unsupported graph schema version"))
        data.terms isa Tuple && data.outputs isa Tuple && data.inputs isa Tuple && data.scalar isa Bool ||
            throw(ArgumentError("invalid graph metadata"))
        terms = map(_read_term,data.terms)
        return TensorGraph(terms,data.outputs,data.inputs;scalar=data.scalar)
    end
    throw(ArgumentError("unknown structural kind $(data.kind)"))
end

"""Restore through validated constructors; never evaluate expressions or executable input."""
function graphfromdata(data)
    graph = _decode(data)
    graph isa TensorGraph || throw(ArgumentError("the root record must be a graph"))
    return graph
end
