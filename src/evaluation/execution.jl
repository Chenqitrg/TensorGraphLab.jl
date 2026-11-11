const _execution_cache = Dict{Any,Function}()
const _execution_lock = ReentrantLock()

"""Release cached contraction functions; existing tensors and graphs are unchanged."""
function clear_evaluation_cache!()
    lock(_execution_lock) do
        empty!(_execution_cache)
    end
    return nothing
end

function _compile(flat::TensorGraph, method::Symbol;
                  tree::Union{Nothing,Integer,Tuple}=nothing, opt::Union{Bool,Expr}=false)
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
    kwargs = opt === false ? () : (Expr(:(=), :opt, opt),)
    contraction = Expr(:macrocall,macroref,LineNumberNode(0),kwargs...,assignment)
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

"""A reusable contraction function with operands in expanded occurrence order."""
struct EvaluationPlan{N<:Tuple,F<:Function}
    nodes::N
    executor::F
end

(plan::EvaluationPlan)(operands::AbstractTensorMap...) = Base.invokelatest(plan.executor, operands...)

# The plan retains the expanded node views so repeated labels, indexed colors
# and adjoints resolve exactly as they did in the compiled expression.
function (plan::EvaluationPlan)(bindings::Coloring)
    return plan(map(node -> _bind(node, bindings), plan.nodes)...)
end

"""
    plan_evaluation(graph::TensorGraph; method=:tensor, opt=false, small_indices=()) -> EvaluationPlan

Compile a contraction without binding tensor data. Pass an iterable of Symbol or Int
labels for `small_indices`. With optimization enabled,
TensorOperations models each lattice index as O(χ). Root-level indices named in
`small_indices` are O(1) and enable optimization automatically. TensorKit checks
the chosen tree for planar execution. Call `plan(bindings)` with a validated
coloring, or pass one tensor per expanded
non-identity occurrence, in order, with all recorded views already applied.
"""
function plan_evaluation(g::TensorGraph; method::Symbol=:tensor, opt::Bool=false, small_indices=())
    flat, small = _planning_graph(g, Tuple(small_indices))
    return _evaluation_plan(flat, method, opt, small)
end

function _planning_graph(g::TensorGraph, small_indices::Tuple)
    expanded, names = _expand_with_names(g, nothing)
    aliases = Dict{IndexLabel,IndexLabel}()
    flat = _suppress_wires(expanded, aliases)
    small = map(small_indices) do index
        haskey(names, index) || throw(ArgumentError("unknown index $index in the outer graph"))
        id = names[index]
        get(aliases, id, id)
    end
    return flat, Tuple(sort!(unique(collect(Int, small))))
end

function _evaluation_plan(flat::TensorGraph, method::Symbol, opt::Bool, small::Tuple{Vararg{Int}})
    method in (:tensor,:planar) || throw(ArgumentError("method must be :tensor or :planar"))
    optimize = opt || !isempty(small)
    key = (method, graphdata(flat), optimize, small)
    executor = lock(_execution_lock) do
        get!(_execution_cache, key) do
            hint = isempty(small) ? optimize : Expr(:call, :!, small...)
            try
                _compile(flat, method; opt=hint)
            catch err
                # The minimum-cost unrestricted tree may cross a planar contour.
                method === :planar && optimize && err isa ArgumentError &&
                    startswith(err.msg, "not a planar diagram expression") || rethrow()
                _compile(flat, method; tree=_planar_cost_tree(flat, small))
            end
        end
    end
    nodes = map(term -> term.object, occurrences(flat))
    return EvaluationPlan(nodes, executor)
end

"""
    evaluate(graph, bindings::Coloring; method=:tensor, opt=false, small_indices=())

Check the coloring, then contract with TensorKit. Generated plans are shared
across compatible graphs. Scalar output and zero-leg map output are distinct.
"""
function evaluate(g::TensorGraph, c::Coloring; method::Symbol=:tensor, opt::Bool=false, small_indices=())
    flat, small = _planning_graph(g, Tuple(small_indices))
    _prepared_spaces(g,c,flat)
    return _evaluation_plan(flat, method, opt, small)(c)
end
