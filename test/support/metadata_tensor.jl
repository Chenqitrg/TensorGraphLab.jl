# This tensor has no numerical storage: static queries must only inspect its space.
struct MetadataTensor{S,M,N} <: AbstractTensorMap{Float64,S,M,N}
    tensor_space::TensorMapSpace{S,M,N}
end
TensorKit.space(t::MetadataTensor) = t.tensor_space
