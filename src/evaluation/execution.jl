const _execution_cache = Dict{Any,Function}()
const _execution_lock = ReentrantLock()

"""Release cached contraction functions; existing tensors and graphs are unchanged."""
function clear_evaluation_cache!()
    lock(_execution_lock) do
        empty!(_execution_cache)
    end
    return nothing
end

function _compile(flat,method; tree=nothing)
    operands = map(_ -> gensym(:operand),occurrences(flat))
    terms = map(occurrences(flat),operands) do term,operand
        m = numout(term.object)
        _tensorref(operand,term.indices[1:m],term.indices[m+1:end])
    end
    output = gensym(:result)
    lhs = isscalar(flat) ? output : _tensorref(output,boundary(flat)...)
    expression(i::Integer) = terms[i]
    expression(pair::Tuple) = Expr(:call, :*, expression(pair[1]), expression(pair[2]))
    rhs = isnothing(tree) ? (length(terms) == 1 ? only(terms) : Expr(:call,:*,terms...)) : expression(tree)
    assignment = Expr(isscalar(flat) ? :(=) : :(:=),lhs,rhs)
    macroref = GlobalRef(TensorKit,method === :tensor ? Symbol("@tensor") : Symbol("@planar"))
    contraction = Expr(:macrocall,macroref,LineNumberNode(0),assignment)
    # Only trusted graph structure enters this AST, never user code or tensor data.
    body = Expr(:block,contraction,output)
    try
        return Core.eval(@__MODULE__,Expr(:->,Expr(:tuple,operands...),body))
    catch error
        # Macro expansion wraps the upstream error in LoadError.
        error isa LoadError && throw(error.error)
        rethrow()
    end
end

"""
    evaluate(graph, bindings::Coloring; method=:tensor)

Contract bound TensorMaps using TensorKit's @tensor (default) or @planar.
Recheck bindings on every call and replay recorded views on current tensor data.
Generated functions are cached by ordered expanded structure and method; tensors
and numerical results are never cached. Scalar intent returns a scalar, while
an empty indexed boundary returns a zero-leg TensorMap.
"""
function evaluate(g::TensorGraph,c::Coloring;method::Symbol=:tensor)
    method in (:tensor,:planar) || throw(ArgumentError("method must be :tensor or :planar"))
    flat,tensors = _prepared(g,c)
    # Canonical graph equality ignores factor order, which planar execution needs.
    key = (method,graphdata(flat))
    executor = lock(_execution_lock) do
        get!(_execution_cache,key) do
            _compile(flat,method)
        end
    end
    return Base.invokelatest(executor,tensors...)
end
