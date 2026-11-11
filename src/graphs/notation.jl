# All index spellings and tensor references are recognized by TensorOperations.
function _product_terms!(terms, ex)
    if TO.istensor(ex)
        push!(terms, ex)
    elseif ex isa Expr && ex.head === :call && ex.args[1] === :* && length(ex.args) >= 3
        foreach(arg -> _product_terms!(terms,arg), ex.args[2:end])
    else
        throw(ArgumentError("@graph requires indexed nodes/subgraphs or their product; sums and scalar factors are unsupported"))
    end
    return terms
end

function _graph_macro(wrap, ex)
    if ex isa Expr && ex.head === :block
        return Expr(:block, (item isa LineNumberNode ? item : _graph_macro(wrap, item) for item in ex.args)...)
    end
    ex isa Expr && (TO.isdefinition(ex) || ex.head === :(=)) ||
        throw(ArgumentError("use @graph G[outputs; inputs] := ... or @graph scalar_name = ..."))
    TO.verifytensorexpr(ex)
    ex = TO.normalizeindices(ex)
    lhs, rhs = TO.getlhs(ex), TO.getrhs(ex)
    scalar = !TO.istensor(lhs)
    if scalar
        lhs isa Symbol && ex.head === :(=) || throw(ArgumentError("scalar graphs require a bare name and ="))
        name, outputs, inputs = lhs, (), ()
    else
        TO.isdefinition(ex) || throw(ArgumentError("use := to define an immutable graph; indexed = is unsupported"))
        name, left, right = TO.decomposetensor(lhs)
        name isa Symbol || throw(ArgumentError("the output graph must be bound to a variable name"))
        outputs, inputs = Tuple(left), Tuple(right)
    end
    _product_terms!(Any[],rhs) # Reject unsupported operations before capturing any objects.
    objects = TO.getinputtensorobjects(rhs)
    variables = Dict(object => gensym(:graph_input) for object in objects)
    bindings = [Expr(:(=),variables[object],esc(object)) for object in objects]
    captured = TO.replacetensorobjects(rhs) do object, left, right
        get(variables,object,object)
    end
    terms = map(_product_terms!(Any[],captured)) do term
        object, left, right = TO.decomposetensor(term)
        :($Occurrence($object, $(QuoteNode(Tuple(left))), $(QuoteNode(Tuple(right)))))
    end
    termtuple = Expr(:tuple,terms...)
    construct = :($TensorGraph($termtuple, $(QuoteNode(outputs)), $(QuoteNode(inputs)); scalar=$scalar))
    scoped = Expr(:let,Expr(:block,bindings...),wrap(construct))
    return :($(esc(name)) = $scoped)
end

"""
    @graph G[outputs; inputs] := A[...] * B[...]
    @graph z = A[i; i]

Capture a structural graph with TensorOperations index syntax and explicit node
bindings. No tensors are evaluated. Products and indexed node views are supported;
each distinct object expression is evaluated once. Output/input partitions must
match the captured objects. Bare-name = records scalar output intent.
"""
macro graph(ex)
    return _graph_macro(identity, ex)
end
