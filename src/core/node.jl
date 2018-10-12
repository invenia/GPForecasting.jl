getfields(x) = getfield.(x, fieldnames(x))

abstract type AbstractNode end

# Extract children and not children.
extract_children(x) = AbstractNode[]
extract_children(x::AbstractNode) = [x]
extract_children(x::AbstractArray{<:AbstractNode}) = x[:]
extract_others(x) = [x]
extract_others(x::AbstractNode) = []
extract_others(x::AbstractArray{<:AbstractNode}) = []
children(x) = vcat(extract_children.(getfields(x))...)
others(x) = vcat(extract_others.(getfields(x))...)

# Reconstruct an instantiated type.
_reconstruct!(field, others, children) = pop!(others)
_reconstruct!(field::AbstractNode, others, children) = pop!(children)
function _reconstruct!(field::AbstractArray{<:AbstractNode}, others, children)
    return reshape([pop!(children) for _ in 1:length(field)], size(field)...)
end
function reconstruct(x, others, children)
    others, children = reverse.(copy.(collect.((others, children))))
    return typeof(x)([_reconstruct!(field, others, children) for field in getfields(x)]...)
end

struct TreeNode <: AbstractNode
    x
    children::Vector{TreeNode}
end

TreeNode(x) = TreeNode(x, TreeNode[])

Base.:(==)(tn1::TreeNode, tn2::TreeNode) = tn1.x == tn2.x && tn1.children == tn2.children

function Base.show(io::IO, tn::TreeNode)
    print(io, "TreeNode(", tn.x)
    if isempty(tn.children)
        print(io, ")")
    else
        print(io, ", [")
        join(io, tn.children, ", ")
        print(io, "])")
    end
end

function map(f, ns::TreeNode...)
    return TreeNode(
        f([n.x for n in ns]...),
        [map.(f, ps...) for ps in zip(getfield.(ns, :children)...)]
    )
end
zip(ns::TreeNode...) = map((xs...) -> xs, ns...)
reduce(op, n::TreeNode) = op(n.x, _reduce(op, n)...)
_reduce(op, n::TreeNode) = [op(p.x, _reduce(op, p)...) for p in n.children]
function reduce(op, ns::TreeNode...)
    return reduce((xs, children...) -> op(xs..., children...), zip(ns...))
end

mutable struct StackedVector
    v
    offset::Integer
end
StackedVector(v) = StackedVector(v, 0)
function get_next!(sv::StackedVector, len::Integer)
    res = sv.v[sv.offset + 1:sv.offset + len]
    sv.offset += len
    return res
end

function flatten(t::TreeNode, res::Vector=Any([]))
    append!(res, t.x)
    foreach(p -> flatten(p, res), t.children)
    return res
end
@unionise interpret(t::TreeNode, v::Vector) = interpret(t, StackedVector(v))
@unionise function interpret(t::TreeNode, sv::StackedVector)
    return TreeNode(get_next!(sv, length(t.x)), [interpret(p, sv) for p in t.children])
end

# This is a "two-dimensional" flattening, where it is assumed that nodes contain `Vector`s
# of items, and we would like to also reconstruct those `Vector`s upon interpretation.
function flatten2(t::TreeNode, res::Vector=[])
    foreach(x -> append!(res, x), t.x)
    foreach(p -> flatten2(p, res), t.children)
    return res
end
@unionise interpret2(t::TreeNode, v::Vector) = interpret2(t, StackedVector(v))
@unionise function interpret2(t::TreeNode, sv::StackedVector)
    return TreeNode(
        [get_next!(sv, length(y)) for y in t.x],
        [interpret2(p, sv) for p in t.children]
    )
end

# Get set for AbstractNode, might have to also dispatch this for Random
tree(m::Union{AbstractNode, Random}) = TreeNode(m, tree.(children(m)))
pack(m::Union{AbstractNode, Random}) = pack.(others(m))
unpack(original::Union{AbstractNode, Random}, data, children::Union{AbstractNode, Random}...) =
    reconstruct(original, unpack.(others(original), data), children)
get(m::Union{AbstractNode, Random}) = flatten2(map(pack, tree(m)), Float64[])
function get(m::Union{AbstractNode, Random}, n::String)
    θ = unwrap.(filter(x -> name(x) == n, flatten(map(others, tree(m)))))
    return length(θ) == 1 ? θ[1] : θ
end
@unionise function set(m::Union{AbstractNode, Random}, θ::Vector)
    t = tree(m)
    return reduce(unpack, t, interpret2(map(pack, t), θ))
end
@unionise function set(m::Union{AbstractNode, Random}, updates::Pair...)
    d, t = Dict(updates...), tree(m)
    update(x) = haskey(d, name(x)) ? set(x, d[name(x)]) : x
    return reduce(unpack, t, map(x -> update.(x), map(others, t)))
end
getindex(m::Union{AbstractNode, Random}, ::Colon) = get(m)
getindex(m::Union{AbstractNode, Random}, n::String) = get(m, n)
getindex(m::Union{AbstractNode, Random}, ns::String...) = get.(m, ns)
