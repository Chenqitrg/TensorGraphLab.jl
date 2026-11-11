"""An occurrence of one color family with a particular binding slot and port signature."""
struct IndexedColor{M,N} <: AbstractTensorNode{M,N}
    node::TensorNode{M,N}
    slot::Int
    function IndexedColor(node::TensorNode{M,N}, slot::Int) where {M,N}
        slot > 0 || throw(ArgumentError("binding slots start at one"))
        new{M,N}(node, slot)
    end
end

base_node(n::IndexedColor) = n
label(n::IndexedColor) = label(n.node)
dualflags(n::IndexedColor) = dualflags(n.node)
_key(n::IndexedColor) = (:indexed, _key(n.node), n.slot)
_nodekey(n::IndexedColor) = (:indexed, _nodekey(n.node), n.slot)
_nodeexpr(n::IndexedColor) = _nodeexpr(n.node)
_replace(n::IndexedColor, replacements::AbstractDict{Symbol,<:AbstractTensorNode}) = get(replacements, label(n), n)
_same_color(a::AbstractTensorNode, b::AbstractTensorNode) = a == b
_same_color(::IndexedColor{M,N}, ::IndexedColor{M,N}) where {M,N} = true
