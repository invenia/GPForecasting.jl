getfields(x) = broadcast(y -> getfield(x, y), fieldnames(typeof(x)))

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
extract_children(x::Tuple) = vcat(extract_children.(x)...)
extract_others(x) = [x]
extract_others(x::AbstractNode) = []
extract_others(x::AbstractArray{<:AbstractNode}) = []
extract_others(x::Tuple) = vcat(extract_others.(x)...)


"""
    children(x) -> Vector{AbstractNode}

Return a vector of all nodes directly referenced by `x`, including those in arrays.
"""
children(x) = vcat(extract_children.(getfields(x))...)

"""
    others(x) -> Vector{Any}

Return all non-node fields (leaves) of `x`, i.e., everything not returned by [`children`](@ref).
"""
others(x) = vcat(extract_others.(getfields(x))...)

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
function _reconstruct!(field::Tuple, others, children)
    return ntuple(length(field)) do i
        _reconstruct!(field[i], others, children)
    end
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
        [_reconstruct!(field, others, children) for field in getfields(x)]...
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

"""
    map(f, ns::TreeNode...) -> TreeNode

Recurses through parallel trees and applies a function to parallel nodes.

# Example

```jldoctest; setup = :(import GPForecasting: TreeNode)
julia> a = TreeNode(1, [TreeNode(2), TreeNode(3)]); map(+, a, a)
TreeNode(2, GPForecasting.TreeNode[TreeNode(4, GPForecasting.TreeNode[]), TreeNode(6, GPForecasting.TreeNode[])])
```
"""
function Base.map(f, ns::TreeNode...)
    return TreeNode(
        f([n.x for n in ns]...),
        [map.(f, ps...) for ps in zip(getfield.(ns, :children)...)]
    )
end

"""
    zip(ns::TreeNode...) -> TreeNode

Recurses through parallel trees and collects parallel nodes into nodes of tuples.
Equivalent to `map(tuple, ns...)`.

# Example

```jldoctest; setup = :(import GPForecasting: TreeNode)
julia> a = TreeNode(1, [TreeNode(2), TreeNode(3)]); zip(a, a)
TreeNode((1, 1), GPForecasting.TreeNode[TreeNode((2, 2), GPForecasting.TreeNode[]), TreeNode((3, 3), GPForecasting.TreeNode[])])
```
"""
Base.Iterators.zip(ns::TreeNode...) = map((xs...) -> xs, ns...)
Base.reduce(op, n::TreeNode) = op(n.x, _reduce(op, n)...)
_reduce(op, n::TreeNode) = [op(p.x, _reduce(op, p)...) for p in n.children]

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
6-element Vector{Int64}:
 1
 2
 3
 4
 5
 6

julia> c = TreeNode([[1, 2], [3, 4]],
               [TreeNode([[5, 6], [7, 8]]), TreeNode([[9, 10], [11, 12]])]); flatten(c)
6-element Vector{Any}:
 [1, 2]
 [3, 4]
 [5, 6]
 [7, 8]
 [9, 10]
 [11, 12]
```
"""
function flatten(t::TreeNode, res::Vector=Any[])
    append!(res, t.x)
    foreach(p -> flatten(p, res), t.children)
    return res
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
6-element Vector{Int64}:
 1
 2
 3
 4
 5
 6

julia> c = TreeNode([[1, 2], [3, 4]],
               [TreeNode([[5, 6], [7, 8]]), TreeNode([[9, 10], [11, 12]])]); flatten2(c)
12-element Vector{Any}:
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
function flatten2(t::TreeNode, res::Vector=[])
    foreach(x -> append!(res, x), t.x)
    foreach(p -> flatten2(p, res), t.children)
    return res
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
    ?? = unwrap.(filter(x -> name(x) == n, flatten(map(others, tree(m)))))
    return length(??) == 1 ? ??[1] : ??
end

"""
    set(m, ??::Vector) -> AbstractNode

Reconstruct the original structure of `m` with the modified parameters from `??`, by first
converting to a `TreeNode`.
"""
@unionise function set(m::Union{AbstractNode, Process}, ??::Vector)
    t = tree(m)
    return reduce(unpack, t, interpret2(map(pack, t), ??))
end

"""
    set(m, updates::Pair...) -> AbstractNode

Reconstruct the original structure of `m`, updating nodes named `k` with the modified
parameters from `v`, for each `k => v` in `updates`.

See also [`set(m, ??::Vector)`](@ref set(m, ::Vector)).
"""
@unionise function set(m::Union{AbstractNode, Process}, updates::Pair...)
    d, t = Dict(updates...), tree(m)
    update(x) = haskey(d, name(x)) ? set(x, d[name(x)]) : x
    return reduce(unpack, t, map(x -> update.(x), map(others, t)))
end
Base.getindex(m::Union{AbstractNode, Process}, ::Colon) = get(m)
Base.getindex(m::Union{AbstractNode, Process}, n::String) = get(m, n)
Base.getindex(m::Union{AbstractNode, Process}, ns::String...) = get.(m, ns)
