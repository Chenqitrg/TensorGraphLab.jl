macro tube(option, ex)
    Meta.isexpr(option, :(=), 2) && option.args[1] === :annular &&
        Meta.isexpr(option.args[2], :tuple) ||
        throw(ArgumentError("use @tube annular=(...) T[outputs; inputs] := ..."))
    annular = QuoteNode(Tuple(map(TO.normalizeindex, option.args[2].args)))
    return _graph_macro(ex) do graph
        :($TubeGraph($graph; annular=$annular))
    end
end

function showgraph(io::IO, tube::TubeGraph; name::Symbol=:T, depth::Union{Int,Nothing}=nothing)
    expression, children = _display_parts(_closed_graph(tube); name, depth, keep=tube.annular)
    for (binding, child) in children
        println(io, "# ", binding, " represents ", child)
    end
    print(io, "@tube annular=")
    Base.show_unquoted(io, Expr(:tuple, tube.annular...))
    print(io, ' ')
    Base.show_unquoted(io, expression)
    println(io)
    return nothing
end
showgraph(t::TubeGraph; kwargs...) = showgraph(stdout, t; kwargs...)
