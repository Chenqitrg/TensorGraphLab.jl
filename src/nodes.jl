"""A node or structural view with `M` output and `N` input ports; never a numerical tensor."""
abstract type AbstractTensorNode{M,N} end

"""
    TensorNode(label::Symbol, (nout, nin))
    TensorNode(label::Symbol, nout, nin, duals::Tuple)
    TensorNode(label::Symbol, tensor::AbstractTensorMap)

An immutable color label and ordered, oriented ports. `duals` lists output ports
followed by input ports in the all-outgoing convention. `true` denotes a formally
dual leg. No tensor, concrete space, dimension or sector data is retained.
The type parameters of `TensorNode{M,N}` count output and input ports separately.

The rank-pair constructor assumes nondual codomain and domain spaces: output
flags are `false` and input flags are `true` in the all-outgoing convention.
Both counts must be nonnegative; zero-leg partitions are supported.

Extraction uses codomain flags followed by **negated domain flags**. Unlike
`isdual(space(t,i))`, this preserves formal input duals for real Cartesian spaces.
Once numerical data has erased arrow history, extraction cannot reconstruct it;
transform the node itself to preserve that history.
"""
struct TensorNode{M,N} <: AbstractTensorNode{M,N}
    label::Symbol
    outduals::NTuple{M,Bool}
    induals::NTuple{N,Bool}
    function TensorNode(label::Symbol, outduals::NTuple{M,Bool}, induals::NTuple{N,Bool}) where {M,N}
        return new{M,N}(label, outduals, induals)
    end
end

function TensorNode(label::Symbol, ranks::Tuple{Int,Int})
    nout, nin = ranks
    nout >= 0 && nin >= 0 || throw(ArgumentError("port counts must be nonnegative"))
    return TensorNode(label, ntuple(_ -> false, nout), ntuple(_ -> true, nin))
end

function TensorNode(label::Symbol, nout::Int, nin::Int, duals::NTuple{L,Bool}) where {L}
    nout >= 0 && nin >= 0 || throw(ArgumentError("port counts must be nonnegative"))
    nout <= L && nin == L - nout ||
        throw(ArgumentError("expected one direction flag per port"))
    return TensorNode(label, ntuple(i -> duals[i], nout), ntuple(i -> duals[nout + i], nin))
end

function TensorNode(name::Symbol, t::AbstractTensorMap)
    outputs = ntuple(i -> isdual(codomain(t)[i]), numout(t))
    inputs = ntuple(i -> !isdual(domain(t)[i]), numin(t))
    return TensorNode(name, outputs, inputs)
end

"""A lazy adjoint whose parent has the opposite input/output partition."""
struct AdjointNode{M,N,T<:AbstractTensorNode{N,M}} <: AbstractTensorNode{M,N}
    parent::T
end

"""A structural port permutation. Prefer `permute(node, indices)`."""
struct PermutedNode{M,N,T<:AbstractTensorNode} <: AbstractTensorNode{M,N}
    parent::T
    indices::Tuple{NTuple{M,Int},NTuple{N,Int}}
    function PermutedNode(node::T, p::TensorKit.Index2Tuple{M,N}) where {M,N,T<:AbstractTensorNode}
        permute(_portspace(node), p)
        return new{M,N,T}(node, p)
    end
end

"""A structural planar transpose. Prefer `transpose(node, indices)`."""
struct TransposedNode{M,N,T<:AbstractTensorNode} <: AbstractTensorNode{M,N}
    parent::T
    indices::Tuple{NTuple{M,Int},NTuple{N,Int}}
    function TransposedNode(node::T, p::TensorKit.Index2Tuple{M,N}) where {M,N,T<:AbstractTensorNode}
        W = _portspace(node)
        permute(W, p) # Check the full bijection before the cyclic-order check.
        transpose(W, p)
        return new{M,N,T}(node, p)
    end
end

const NodeView = Union{AdjointNode,PermutedNode,TransposedNode}
Base.parent(node::NodeView) = node.parent

"""Return the underlying basic node, removing all structural views."""
base_node(node::TensorNode) = node
base_node(node::NodeView) = base_node(parent(node))

"""Return the color label, unchanged by structural transformations."""
label(node::AbstractTensorNode) = base_node(node).label

TensorKit.numout(::AbstractTensorNode{M,N}) where {M,N} = M
TensorKit.numin(::AbstractTensorNode{M,N}) where {M,N} = N
TensorKit.numind(::AbstractTensorNode{M,N}) where {M,N} = M + N

"""Return formal dual flags in output-then-input order, including for real tensors."""
dualflags(node::TensorNode) = (node.outduals..., node.induals...)
function dualflags(node::Union{PermutedNode,TransposedNode})
    original = dualflags(parent(node))
    indices = (node.indices[1]..., node.indices[2]...)
    return map(i -> original[i], indices)
end
function dualflags(node::AdjointNode)
    original = dualflags(parent(node))
    indices = (_inindices(parent(node))..., _outindices(parent(node))...)
    return map(i -> !original[i], indices)
end

_outindices(node) = ntuple(identity, numout(node))
_inindices(node) = ntuple(i -> numout(node) + i, numin(node))

"""Record conjugation, exchange input/output groups, and reverse formal arrows."""
Base.adjoint(node::AbstractTensorNode) = AdjointNode(node)
Base.adjoint(node::AdjointNode) = parent(node)

# Temporary unit spaces only delegate port-permutation validation to TensorKit.
# They carry no user spaces/data and are never stored in a node or view.
function _portspace(node::AbstractTensorNode)
    outputs = ProductSpace{ComplexSpace}(ntuple(_ -> ℂ^1, numout(node)))
    inputs = ProductSpace{ComplexSpace}(ntuple(_ -> ℂ^1, numin(node)))
    return outputs ← inputs
end

"""Record TensorKit's ordered port permutation without a numerical operation."""
function TensorKit.permute(node::AbstractTensorNode, p::TensorKit.Index2Tuple)
    p == (_outindices(node), _inindices(node)) && return node
    return PermutedNode(node, p)
end

"""
    transpose(node, (outindices, inindices))
    transpose(node)

Record a planar cyclic rearrangement, without conjugation. The default reverses
the order of both groups and exchanges them, following TensorKit. Legs retain
their formal arrows as they move; this is different from adjoint.
"""
function Base.transpose(
        node::AbstractTensorNode,
        p::TensorKit.Index2Tuple = (reverse(_inindices(node)), reverse(_outindices(node))))
    p == (_outindices(node), _inindices(node)) && return node
    return TransposedNode(node, p)
end

_key(node::TensorNode) = (:node, node.label, node.outduals, node.induals)
_key(node::AdjointNode) = (:adjoint, _key(parent(node)))
_key(node::PermutedNode) = (:permute, _key(parent(node)), node.indices)
_key(node::TransposedNode) = (:transpose, _key(parent(node)), node.indices)
Base.:(==)(a::AbstractTensorNode, b::AbstractTensorNode) = _key(a) == _key(b)
Base.isequal(a::AbstractTensorNode, b::AbstractTensorNode) = isequal(_key(a), _key(b))
Base.hash(node::AbstractTensorNode, seed::UInt) = hash(_key(node), seed)

function Base.show(io::IO, node::TensorNode)
    print(io, "TensorNode(", repr(label(node)), ", ", numout(node), ", ",
        numin(node), ", ", dualflags(node), ")")
end
function Base.show(io::IO, node::AdjointNode)
    print(io, "adjoint(")
    show(io, parent(node))
    print(io, ")")
end
function _show_transform(io::IO, name::String, node)
    print(io, name, "(")
    show(io, parent(node))
    print(io, ", ", node.indices, ")")
end
Base.show(io::IO, node::PermutedNode) = _show_transform(io, "permute", node)
Base.show(io::IO, node::TransposedNode) = _show_transform(io, "transpose", node)
