export ▷, Kernel, EQ, ConstantKernel, ScaledKernel, StretchedKernel, SumKernel, set,
    DiagonalKernel, PosteriorKernel, MA, ∿, periodicise, stretch, RQ, PeriodicKernel,
    SpecifiedQuantityKernel, ←, hourly_cov, BinaryKernel, ZeroKernel, isMulti,
    SimilarHourKernel

# Default kernel behaviour:
"""
    tree(k::Kernel) -> TreeNode

Build tree structure for kernel `k` and its children.
"""
tree(k::Kernel) = TreeNode(k, tree.(children(k)))

pack(k::Kernel) = pack.(others(k))
unpack(original::Kernel, data, children::Kernel...) =
    reconstruct(original, unpack.(others(original), data), children)

var(k::Kernel, x) = [k(xx) for xx in x]

size(k::Kernel, i::Int) = i < 1 ? BoundsError() : 1

"""
    hourly_cov(k::Kernel, x)

Return a sparse matrix with only the variances (at the diagonal).

    hourly_cov(k::MultiOutputKernel, x)

Return a block-diagonal sparse matrix with the covariances between all outputs for each
given input value, i.e., does not compute covariances for different values of `x`.
"""
hourly_cov(k::Kernel, x) = spdiagm(var(k, x))

"""
    isMulti(k::Kernel)

True if `k` is a multi output kernel, false otherwise. This is useful because chains of
`SumKernel`s and/or `ScaledKernel`s may hide such information.
"""
isMulti(k::Kernel) = false

"""
    PosteriorKernel <: Kernel

Posterior kernel for a GP.

# Fields:
- `k`: prior kernel.
- `x`: points over which `k` was conditioned.
- `U`: Cholesky decomposition of the covariance matrix.

# Methods:
    (k::PosteriorKernel)(x, y)

Build covariance matrix between `x` and `y`.

    (k::PosteriorKernel)(x) = k(x, x)
"""
type PosteriorKernel <: Kernel
    k::Kernel
    x
    U
    PosteriorKernel(k, x, U) = new(k, Fixed(x), Fixed(U))
end
show(io::IO, k::PosteriorKernel) = print(io, "Posterior($(k.k))")
isMulti(k::PosteriorKernel) = isMulti(k.k)
function (k::PosteriorKernel)(x)
    U = unwrap(k.U)
    xd = unwrap(k.x)
    z = k.k(x, xd) / U
    return k.k(x) .- z * z'
end
function (k::PosteriorKernel)(x, y)
    U = unwrap(k.U)
    xd = unwrap(k.x)
    return k.k(x, y) .- (k.k(x, xd) / U) * (U' \ k.k(xd, y))
end
function (k::PosteriorKernel)(x::Real, y::Real)
    U = unwrap(k.U)
    xd = unwrap(k.x)
    return (k.k(x, y) .- (k.k(x, xd) / U) * (U' \ k.k(xd, y)))[1, 1]
end
function (k::PosteriorKernel)(x::Real)
    U = unwrap(k.U)
    xd = unwrap(k.x)
    z = k.k(x, xd) / U
    return (k.k(x) .- z * z')[1, 1]
end

"""
    EQ <: Kernel

Squared exponential kernel. Computes exp((-1/2) * |x - x′|²).
"""
type EQ <: Kernel end
(::EQ)(x, y) = exp.((-0.5) .* sq_pairwise_dist(x, y))
(k::EQ)(x) = k(x, x)
show(io::IO, k::EQ) = print(io, "EQ()")

"""
    RQ <: Kernel

Rational quadratic kernel. Computes (1 + ((x - x′)² / (2α)))^α.
"""
type RQ <: Kernel
    α
    RQ(α) = isconstrained(α) ? new(α) : new(Positive(α))
end
(k::RQ)(x, y) = (1.0 .+ (sq_pairwise_dist(x, y) ./ (2.0 * unwrap(k.α)))) .^ (-unwrap(k.α))
(k::RQ)(x) = k(x, x)
show(io::IO, k::RQ) = print(io, "RQ($(k.α))")

type SimilarHourKernel <: Kernel
    hdeltas::Fixed{Int}
    coeffs
    function SimilarHourKernel(hdeltas::Union{Fixed{Int}, Int}, coeffs)
        unwrap(hdeltas) > 24 && throw(ArgumentError("Can't have more than 24 hour deltas."))
        unwrap(hdeltas) < 1 && throw(ArgumentError("Need at least one hour delta."))
        length(unwrap(coeffs)) != unwrap(hdeltas) && throw(
            DimensionMismatch(
                "The number of coefficients must be the same as the number of hour deltas."
            )
        )
        hd = isconstrained(hdeltas) ? hdeltas : Fixed(hdeltas)
        cs = isconstrained(coeffs) ? coeffs : Positive(coeffs)
        return new(hd, cs)
    end
end
function (k::SimilarHourKernel)(x, y)
    δ(x) = isapprox(x, 0.0, atol=1e-15) ? 1 : 0
    d = pairwise_dist(x, y) .% 24
    cs = unwrap(k.coeffs)
    hd = unwrap(k.hdeltas)
    K = cs[1] * δ.(d)
    return length(cs) > 1 ?
        K .+ sum([(cs[i + 1] * (δ.(d .- (24 - i)) + δ.(d .- i))) for i in 1:(hd - 1)]) :
        K
end
(k::SimilarHourKernel)(x) = k(x, x)
function show(io::IO, k::SimilarHourKernel)
    cs = unwrap(k.coeffs)
    ds = "$(cs[1])*δ(0)"
    for i in 1:(unwrap(k.hdeltas) - 1)
        ds *= " + $(cs[i + 1])*δ($i)"
    end
    print(io, ds)
end


"""
    MA <: Kernel

Matérn kernel. Implemented only for ν in [1/2, 3/2, 5/2].
"""
type MA <: Kernel
    ν::Fixed
end
MA(n::Real) = MA(Fixed(n))
function (k::MA)(x, y)
    d = pairwise_dist(x, y)
    if unwrap(k.ν) ≈ 1/2
        return exp.(-d)
    elseif unwrap(k.ν) ≈ 3/2
        return (1 + √3 .* d) .* exp.(-√3 .* d)
    elseif unwrap(k.ν) ≈ 5/2
        return (1 + √5 .* d + 5/3 .* d.^2) .* exp.(-√5 .* d)
    else
        throw(ArgumentError("$(unwrap(k.ν)) is not a supported value for Matérn kernels."))
    end
end
(k::MA)(x) = k(x, x)
show(io::IO, k::MA) = print(io, "MA($(k.ν))")

"""
    BinaryKernel <: Kernel

Kernel for binary inputs. Has three possible outcomes: Θ₁ if x = y = 1, Θ₂ if x = y = 0 and
Θ₃ if x ≠ y. Naturally, this only accepts unidimensional inputs.
"""
type BinaryKernel <: Kernel
    Θ₁
    Θ₂
    Θ₃
end
function (k::BinaryKernel)(x, y)
    return unwrap(k.Θ₁) .* (x .≈ y' .≈ 1) +
        unwrap(k.Θ₂) .* (x .≈ y' .≈ 0) +
        unwrap(k.Θ₃) .* (x .!= y')
end
(k::BinaryKernel)(x) = k(x, x)
function BinaryKernel(a::Real, b::Real, c::Real)
    pa = Positive(a)
    pb = Positive(b)
    return BinaryKernel(
        pa,
        pb,
        Bounded(
            c,
            DynamicBound((x, y) -> -sqrt(x * y), [pa, pb]),
            DynamicBound((x, y) -> sqrt(x * y), [pa, pb]),
        )
    )
end
BinaryKernel(a::Real, b::Real) = BinaryKernel(Positive(a), Positive(b), Fixed(0))

"""
    ScaledKernel <: Kernel

Result from the multiplication of a `Kernel` by a number or `Parameter`.
Scales the kernel variance.
"""
type ScaledKernel <: Kernel
    scale
    k::Kernel
end
isMulti(k::ScaledKernel) = isMulti(k.k)
function (*)(x, k::Kernel)
    return isconstrained(x) ?
        ScaledKernel(x, k) :
        (unwrap(x) ≈ zero(unwrap(x)) ? ZeroKernel() : ScaledKernel(Positive(x), k))
end
function (*)(x, k::ScaledKernel)
    return ScaledKernel(
        isconstrained(x) ? x * unwrap(k.scale) : Positive(x * unwrap(k.scale)),
        k.k
    )
end
(*)(k::Kernel, x) = (*)(x, k::Kernel)
function (+)(k1::Kernel, k2::ScaledKernel)
    return unwrap(k2.scale) ≈ zero(unwrap(k2.scale)) ? k1 : SumKernel(k1, k2)
end
function (+)(k1::ScaledKernel, k2::Kernel)
    return unwrap(k1.scale) ≈ zero(unwrap(k1.scale)) ? k2 : SumKernel(k1, k2)
end
function (+)(k1::ScaledKernel, k2::ScaledKernel)
    if unwrap(k1.scale) ≈ zero(unwrap(k1.scale))
        return k2
    elseif unwrap(k2.scale) ≈ zero(unwrap(k2.scale))
        return k1
    else
        return SumKernel(k1, k2)
    end
end
(k::ScaledKernel)(x, y) = unwrap(k.scale) .* k.k(x, y)
(k::ScaledKernel)(x) = k(x, x)
show(io::IO, k::ScaledKernel) = print(io, "($(k.scale) * $(k.k))")

"""
    StretchedKernel <: Kernel

Represent any `Kernel` with length scale stretched to `stretch`.
"""
type StretchedKernel <: Kernel
    stretch
    k::Kernel
end
isMulti(k::StretchedKernel) = isMulti(k.k)

"""
    k ▷ l
    stretch(k, l)

Stretch `Kernel` k's length scale by l.
"""
stretch(k::Kernel, x) = StretchedKernel(isconstrained(x) ? x : Positive(x), k)
(▷)(k::Kernel, x) = stretch(k, x)
function stretch(k::StretchedKernel, x)
    return StretchedKernel(
        isconstrained(x) ? x .* unwrap(k.stretch) : Positive(x .* unwrap(k.stretch)),
        k.k
    )
end
function (k::StretchedKernel)(x, y)
    lscale = unwrap(k.stretch)'

    # This condition should only be met in case the input space got extended by `periodicise`
    # If the user forces this to trigger by feeding a length scale with the wrong length,
    # that is his fault.
    length(lscale) > 1 && length(lscale) == size(x, 2) / 2 &&
    return k.k(x ./ hcat(lscale, lscale), y ./ hcat(lscale, lscale))

    return k.k(x ./ lscale, y ./ lscale)
end
(k::StretchedKernel)(x) = k(x, x)
show(io::IO, k::StretchedKernel) = print(io, "($(k.k) ▷ $(k.stretch))")

"""
    SumKernel <: Kernel

Kernel built by adding two kernels, `k1` and `k2`.
"""
type SumKernel <: Kernel
    k1::Kernel
    k2::Kernel
end
(+)(k1::Kernel, k2::Kernel) = SumKernel(k1, k2)
(k::SumKernel)(x, y) = k.k1(x, y) .+ k.k2(x, y)
(k::SumKernel)(x) = k(x, x)
show(io::IO, k::SumKernel) = print(io, "($(k.k1) + $(k.k2))")
isMulti(k::SumKernel) = isMulti(k.k1) || isMulti(k.k2)

"""
    ProductKernel <: Kernel

Kernel built by multiplying two kernels, `k1` and `k2`.
"""
type ProductKernel <: Kernel
    k1::Kernel
    k2::Kernel
end
(*)(k1::Kernel, k2::Kernel) = ProductKernel(k1, k2)
function (*)(k1::Kernel, k2::ScaledKernel)
    return unwrap(k2.scale) ≈ zero(unwrap(k2.scale)) ? Kernel(0) : ProductKernel(k1, k2)
end
function (*)(k1::ScaledKernel, k2::Kernel)
    return unwrap(k1.scale) ≈ zero(unwrap(k1.scale)) ? Kernel(0) : ProductKernel(k1, k2)
end
function (*)(k1::ScaledKernel, k2::ScaledKernel)
    if unwrap(k1.scale) ≈ zero(unwrap(k1.scale))
        return Kernel(0)
    elseif unwrap(k2.scale) ≈ zero(unwrap(k2.scale))
        return Kernel(0)
    else
        return ProductKernel(k1, k2)
    end
end
(k::ProductKernel)(x, y) = k.k1(x, y) .* k.k2(x, y)
(k::ProductKernel)(x) = k(x, x)
show(io::IO, k::ProductKernel) = print(io, "($(k.k1) * $(k.k2))")
isMulti(k::ProductKernel) = isMulti(k.k1) || isMulti(k.k2)

"""
    PeriodicKernel <: Kernel

Kernel built by defining a period `T` for kernel `k`.
"""
type PeriodicKernel <: Kernel
    T
    k::Kernel
end
function (k::PeriodicKernel)(x, y)
    px = [cos.(2π .* x ./ unwrap(k.T)') sin.(2π .* x ./ unwrap(k.T)')]
    py = [cos.(2π .* y ./ unwrap(k.T)') sin.(2π .* y ./ unwrap(k.T)')]
    return k.k(px, py)
end
(k::PeriodicKernel)(x) = k(x, x)
show(io::IO, k::PeriodicKernel) = print(io, "($(k.k) ∿ $(k.T))")
isMulti(k::PeriodicKernel) = isMulti(k.k)

"""
    periodicise(k::Kernel, l::Real)

Turn kernel `k` into a periodic kernel of period `l`.
"""
periodicise(k::Kernel, l) = PeriodicKernel(isconstrained(l) ? l : Positive(l), k)
# (∿)(k::Kernel, l::Real) = periodicise(k, l) # Charachter seems to be problematic

"""
    SpecifiedQuantityKernel <: Kernel

A kernel `k` that acts on the column `col` of a dataframe. Allows for input selection.
"""
type SpecifiedQuantityKernel <: Kernel
    col::Fixed
    k::Kernel
end
(←)(k::Kernel, s::Symbol) = SpecifiedQuantityKernel(Fixed(s), k)
function (k::SpecifiedQuantityKernel)(x::DataFrame, y::DataFrame)
    return k.k(disallowmissing(x[unwrap(k.col)]), disallowmissing(y[unwrap(k.col)]))
end
(k::SpecifiedQuantityKernel)(x::DataFrame) = k(x, x)
show(io::IO, k::SpecifiedQuantityKernel) = print(io, "($(k.k) ← $(k.col))")
isMulti(k::SpecifiedQuantityKernel) = isMulti(k.k)

"""
    ConstantKernel <: Kernel

Kernel that returns 1.0 for every pair of points.
"""
type ConstantKernel <: Kernel end
(k::ConstantKernel)(x, y) = ones(Float64, size(x, 1), size(y, 1))
(k::ConstantKernel)(x) = k(x, x)
(k::ConstantKernel)(x::Real, y::Real) = 1.0
function (+)(k::Kernel, x)
    return isconstrained(x) ?
        SumKernel(k, x * ConstantKernel()) :
        (unwrap(x) ≈ zero(unwrap(x)) ? k : SumKernel(k, x * ConstantKernel()))
end
(+)(x, k::Kernel) = (+)(k::Kernel, x)
convert(::Type{Kernel}, x::Real) = x ≈ 0.0 ? ZeroKernel() : Fixed(x) * ConstantKernel()
show(io::IO, k::ConstantKernel) = print(io, "𝟏")

"""
    ZeroKernel <: Kernel

Zero kernel. Returns zero.
"""
type ZeroKernel <: Kernel; end
(::ZeroKernel)(x, y) = zeros(size(x, 1), size(y, 1))
(k::ZeroKernel)(x) = k(x, x)
(+)(k::Kernel, z::ZeroKernel) = k
(+)(z::ZeroKernel, k::Kernel) = k + z
(+)(z::ZeroKernel, k::ZeroKernel) = z
(+)(z::ZeroKernel, k::ScaledKernel) = k
(+)(k::ScaledKernel, z::ZeroKernel) = z + k
(*)(k::Kernel, z::ZeroKernel) = z
(*)(z::ZeroKernel, k::Kernel) = k * z
(*)(z::ZeroKernel, k::ZeroKernel) = z
(*)(z::ZeroKernel, k::ScaledKernel) = z
(*)(k::ScaledKernel, z::ZeroKernel) = z * k
(*)(x, z::ZeroKernel) = z
(*)(z::ZeroKernel, x) = x * z
show(io::IO, z::ZeroKernel) = print(io, "𝟎")

"""
    DiagonalKernel <: Kernel

Diagonal kernel. Has unitary variance.
"""
type DiagonalKernel <: Kernel end
function (::DiagonalKernel)(x, y)
    xl = [x[i, :] for i in 1:size(x, 1)]
    yl = [y[i, :]' for i in 1:size(y, 1)]
    return float.(isapprox.(float.(xl), float.(yl')))
end
(k::DiagonalKernel)(x::Number, y) = k([x], y)
(k::DiagonalKernel)(x, y::Number) = k(x, [y])
(k::DiagonalKernel)(x::Number, y::Number) = k([x], [y])[1, 1]
(k::DiagonalKernel)(x) = k(x, x)
show(io::IO, k::DiagonalKernel) = print(io, "δₓ")

zero(::Kernel) = ZeroKernel()
zero(::Type{GPForecasting.Kernel}) = ZeroKernel()
