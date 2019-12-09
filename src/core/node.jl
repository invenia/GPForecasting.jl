getfields(x) = (getfield(x, i) for i in 1:nfields(x))

abstract type AbstractNode end

"""
    Process <: AbstractNode

Abstract supertype for all stochastic processes.
"""
abstract type Process <: AbstractNode end

# make nodes act as a scalar in broadcasting
Base.broadcastable(tn::AbstractNode) = Ref(tn)

# Extract children and not children.
extract_children(x) = AbstractNode[]
extract_children(x::AbstractNode) = [x]
extract_children(x::AbstractArray{<:AbstractNode}) = x[:]
extract_others(x) = [x]
extract_others(x::AbstractNode) = []
extract_others(x::AbstractArray{<:AbstractNode}) = []

"""
    children(x) -> Vector{AbstractNode}

Return a vector of all nodes directly referenced by `x`, including those in arrays.
"""
children(x) = reduce(vcat, (extract_children(field) for field in getfields(x)), init=AbstractNode[])

"""
    others(x) -> Vector{Any}

Return all non-node fields (leaves) of `x`, i.e., everything not returned by [`children`](@ref).
"""
others(x) = reduce(vcat, (extract_others(field) for field in getfields(x)), init=Any[])

"""
    create_instance(T::Type, args...)

Create an instance of type `T` using arguments `args`.
"""
create_instance(T::Type, args...) = T(args...)

# Reconstruct an instantiated type.
_reconstruct!(field, others, children) = pop!(others)
_reconstruct!(field::AbstractNode, others, children) = pop!(children)
function _reconstruct!(field::AbstractArray{<:AbstractNode}, others, children)
    return reshape([pop!(children) for _ in 1:length(field)], size(field)...)
end

"""
    reconstruct(x::T, others, children) -> T

Given an example instance of `T`, construct a `T` from vectors similar to those generated
by calling [`children`](@ref) and [`others`](@ref) on an instance of `T`.
Used in [`set`](@ref)/[`unpack`](@ref) to reconstruct types from nodes after parameters
have been updated.
"""
function reconstruct(x, others, children)
    others, children = reverse.(copy.(collect.((others, children))))

    return create_instance(
        typeof(x),
        (_reconstruct!(field, others, children) for field in getfields(x))...
    )
end

"""
    TreeNode(x, children=TreeNode[])

A recursive tree structure for `AbstractNode`.
"""
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

"""
    map(f, ns::TreeNode...) -> TreeNode

Recurses through parallel trees and applies a function to parallel nodes.

# Example

```jldoctest; setup = :(import GPForecasting: TreeNode)
julia> a = TreeNode(1, [TreeNode(2), TreeNode(3)]); map(+, a, a)
TreeNode(2, [TreeNode(4), TreeNode(6)])
```
"""
function Base.map(f, ns::TreeNode...)
    return TreeNode(
        f((n.x for n in ns)...),
        TreeNode[map.(f, ps...) for ps in zip((n.children for n in ns)...)]
    )
end

"""
    zip(ns::TreeNode...) -> TreeNode

Recurses through parallel trees and collects parallel nodes into nodes of tuples.
Equivalent to `map(tuple, ns...)`.

# Example

```jldoctest; setup = :(import GPForecasting: TreeNode)
julia> a = TreeNode(1, [TreeNode(2), TreeNode(3)]); zip(a, a)
TreeNode((1, 1), [TreeNode((2, 2)), TreeNode((3, 3))])
```
"""
Base.Iterators.zip(ns::TreeNode...) = map(tuple, ns...)
Base.reduce(op, n::TreeNode) = op(n.x, _reduce(op, n)...)
_reduce(op, n::TreeNode) = (op(p.x, _reduce(op, p)...) for p in n.children)

"""
    reduce(f, ns::TreeNode...) -> TreeNode

Calls [`zip`](@ref zip(::Vararg{TreeNode})) on the `TreeNode` arguments, then applies `f` to
the splatted root node data and the non-splatted results of `reduce(f, child)` for each
child node.

Unlike other `reduce` methods, the function passed in must expect varargs and may have to
handle both splatted and non-splatted arguments. This method is likely only useful for
implementing `set`.

# Examples

```jldoctest; setup = :(import GPForecasting: TreeNode)
julia> a = TreeNode(1, [TreeNode(2), TreeNode(3)]); reduce(+, a, a)
12

julia> reduce(tuple, a)
(1, (2,), (3,))

julia> reduce(tuple, a, a)
(1, 1, (2, 2), (3, 3))
```
"""
function Base.reduce(op, ns::TreeNode...)
    return reduce((xs, children...) -> op(xs..., children...), zip(ns...))
end

"""
    StackedVector(vector)

A vector representing multiple concatenated vector variables, which can be extracted one by
one using [`get_next!`](@ref) and the length to extract.
`StackedVector` keeps track of the current extraction position in its `offset` field.
"""
mutable struct StackedVector
    v
    offset::Integer
end
StackedVector(v) = StackedVector(v, 0)

"""
    get_next!(sv::StackedVector, len::Integer) -> Vector

Retrieve `len` elements from `sv`, starting at `sv`'s `offset`, then increasing the
`offset`.
"""
function get_next!(sv::StackedVector, len::Integer)
    res = sv.v[sv.offset + 1:sv.offset + len]
    sv.offset += len
    return res
end

"""
    flatten(t::TreeNode, result::Vector=[]) -> Vector

Performs a pre-order depth-first traversal of `t`, appending each node's data to `result`,
then return `result`.

# Examples

```jldoctest; setup = :(import GPForecasting: TreeNode, flatten)
julia> b = TreeNode([1, 2], [TreeNode([3, 4]), TreeNode([5, 6])]); flatten(b, Int64[])
6-element Array{Int64,1}:
 1
 2
 3
 4
 5
 6

julia> c = TreeNode([[1, 2], [3, 4]],
               [TreeNode([[5, 6], [7, 8]]), TreeNode([[9, 10], [11, 12]])]); flatten(c)
6-element Array{Any,1}:
 [1, 2]
 [3, 4]
 [5, 6]
 [7, 8]
 [9, 10]
 [11, 12]
```
"""
function flatten(t::TreeNode, result::Vector=[])
    append!(result, t.x)
    foreach(p -> flatten(p, result), t.children)
    return result
end

"""
    interpret(t::TreeNode, v::Vector) -> TreeNode

Returns a tree parallel to `t` where corresponding parameters have been extracted from `v`
and reconstructed into their original node forms.
`interpret` is the inverse of [`flatten`](@ref flatten(::TreeNode)).
"""
@unionise interpret(t::TreeNode, v::Vector) = interpret(t, StackedVector(v))
@unionise function interpret(t::TreeNode, sv::StackedVector)
    return TreeNode(get_next!(sv, length(t.x)), [interpret(p, sv) for p in t.children])
end

# This is a "two-dimensional" flattening, where it is assumed that nodes contain `Vector`s
# of items, and we would like to also reconstruct those `Vector`s upon interpretation.
"""
    flatten2(t::TreeNode, result::Vector=[]) -> Vector

Performs a pre-order depth-first traversal of `t`, appending each element of each node's
data to `result`, then return `result`.

# Examples

```jldoctest; setup = :(import GPForecasting: TreeNode, flatten2)
julia> b = TreeNode([1, 2], [TreeNode([3, 4]), TreeNode([5, 6])]); flatten2(b, Int64[])
6-element Array{Int64,1}:
 1
 2
 3
 4
 5
 6

julia> c = TreeNode([[1, 2], [3, 4]],
               [TreeNode([[5, 6], [7, 8]]), TreeNode([[9, 10], [11, 12]])]); flatten2(c)
12-element Array{Any,1}:
  1
  2
  3
  4
  5
  6
  7
  8
  9
 10
 11
 12
```
"""
function flatten2(t::TreeNode, result::Vector=[])
    foreach(x -> append!(result, x), t.x)
    foreach(p -> flatten2(p, result), t.children)
    return result
end

"""
    interpret2(t::TreeNode, v::Vector) -> TreeNode

Returns a tree parallel to `t` where corresponding parameters have been extracted from `v`
and reconstructed into their original node forms.
`interpret2` is the inverse of [`flatten2`](@ref flatten2(::TreeNode)).
"""
@unionise interpret2(t::TreeNode, v::Vector) = interpret2(t, StackedVector(v))
@unionise function interpret2(t::TreeNode, sv::StackedVector)
    return TreeNode(
        [get_next!(sv, length(y)) for y in t.x],
        [interpret2(p, sv) for p in t.children]
    )
end

# Get set for AbstractNode, might have to also dispatch this for Process
"""
    tree(m) -> TreeNode

Construct an explicit tree structure from a node which references other nodes.
"""
tree(m::Union{AbstractNode, Process}) = TreeNode(m, tree.(children(m)))

"""
    pack(m) -> Vector{Vector}

Get packed parameters for a node, by packing each element of `others(m)`.
This is used by [`get`](@ref get(::AbstractNode)), where the result is flattened into a
single vector.

`pack` and [`unpack`](@ref) are usually defined specially for a given node type.
"""
pack(m::Union{AbstractNode, Process}) = pack.(others(m))

"""
    unpack(original::T, data, children...) -> T

Recursively (depth-first) reconstruct a node/tree from a similar node/tree containing
packed leaves.

[`pack`](@ref) and `unpack` are usually defined specially for a given node type.
"""
unpack(original::Union{AbstractNode, Process}, data, children::Union{AbstractNode, Process}...) =
    reconstruct(original, unpack.(others(original), data), children)

"""
    get(m) -> Vector{Float64}

Make a node into a tree and extract all parameters into a single vector for optimization.
"""
Base.get(m::Union{AbstractNode, Process}) = flatten2(map(pack, tree(m)), Float64[])

"""
    get(m, n::String) -> Vector{Float64}

Make a node into a tree and extract all parameters named (the value of) `n` into a single
vector for optimization.
If there is only one element in the vector, return the first element (this should be
changed).
It is likely that this function is used when there is only one expected element.
"""
function Base.get(m::Union{AbstractNode, Process}, n::String)
    θ = unwrap.(filter(x -> name(x) == n, flatten(map(others, tree(m)))))
    return length(θ) == 1 ? θ[1] : θ
end

"""
    set(m, θ::Vector) -> AbstractNode

Reconstruct the original structure of `m` with the modified parameters from `θ`, by first
converting to a `TreeNode`.
"""
@unionise function set(m::Union{AbstractNode, Process}, θ::Vector)
    t = tree(m)
    return reduce(unpack, t, interpret2(map(pack, t), θ))
end

"""
    set(m, updates::Pair...) -> AbstractNode

Reconstruct the original structure of `m`, updating nodes named `k` with the modified
parameters from `v`, for each `k => v` in `updates`.

See also [`set(m, θ::Vector)`](@ref set(m, ::Vector)).
"""
@unionise function set(m::Union{AbstractNode, Process}, updates::Pair...)
    d, t = Dict(updates...), tree(m)
    update(x) = haskey(d, name(x)) ? set(x, d[name(x)]) : x
    return reduce(unpack, t, map(x -> update.(x), map(others, t)))
end
Base.getindex(m::Union{AbstractNode, Process}, ::Colon) = get(m)
Base.getindex(m::Union{AbstractNode, Process}, n::String) = get(m, n)
Base.getindex(m::Union{AbstractNode, Process}, ns::String...) = get.(m, ns)
