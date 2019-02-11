#########################################################
# Default kernel behaviour:
# k(x) = k(x, x), as a definition for k(x).
# isa(k(x, y), AbstractMatrix) == true, for all x, y.
# k(x, y) == transpose(k(y, x))
#########################################################

Statistics.var(k::Kernel, x) = [k(x[i, :])[1] for i in 1:size(x, 1)]
Statistics.var(k::Kernel, x::Vector{Input}) = reduce(vcat, broadcast(c -> var(k, c), x))
function Statistics.var(k::Kernel, x::AbstractDataFrame)
    return [k(DataFrame(r))[1] for r in eachrow(x)]
end

Base.size(k::Kernel, i::Int) = i < 1 ? BoundsError() : 1

"""
    hourly_cov(k::Kernel, x)

Return a sparse matrix with only the variances (at the diagonal).

    hourly_cov(k::MultiOutputKernel, x)

Return a block-diagonal sparse matrix with the covariances between all outputs for each
given input value, i.e., does not compute covariances for different values of `x`.
"""
hourly_cov(k::Kernel, x) = sparse(Diagonal(var(k, x)))

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
mutable struct PosteriorKernel <: Kernel
    k::Kernel
    x
    U
    PosteriorKernel(k, x, U) = new(k, Fixed(x), Fixed(U))
end
Base.show(io::IO, k::PosteriorKernel) = print(io, "Posterior($(k.k))")
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

"""
"""
mutable struct TitsiasPosteriorKernel <: Kernel
    k::Kernel
    Xm
    Uz
    Umm
    TitsiasPosteriorKernel(k, Xm, Uz, Umm) = new(k, Fixed(Xm), Fixed(Uz), Fixed(Umm))
end

function (k::TitsiasPosteriorKernel)(x)
    Xm = unwrap(k.Xm)
    Uz = unwrap(k.Uz)
    Umm = unwrap(k.Umm)

    Kx = k.k(x)
    Kmx = k.k(Xm, x)
    sqrt₁ = Umm' \ Kmx
    sqrt₂ = Uz' \ Kmx
    return Kx .- (sqrt₁' * sqrt₁) .+ (sqrt₂' * sqrt₂)
end

function (k::TitsiasPosteriorKernel)(x, y)

end


"""
    EQ <: Kernel

Squared exponential kernel. Computes exp((-1/2) * |x - x′|²).
"""
struct EQ <: Kernel end
(::EQ)(x, y) = exp.((-0.5) .* sq_pairwise_dist(x, y))
(k::EQ)(x) = k(x, x)
Base.show(io::IO, k::EQ) = print(io, "EQ()")

const ArrayOrReal = Union{Wrapped{<:AbstractArray{<:Real}}, Wrapped{<:Real}}

"""
    RQ <: Kernel

Rational quadratic kernel. Computes (1 + ((x - x′)² / (2α)))^α.
"""
mutable struct RQ <: Kernel
    α
    RQ(α) = isconstrained(α) ? new(α) : new(Positive(α))
end
function (k::RQ)(x::ArrayOrReal, y::ArrayOrReal)
    return (1.0 .+ (sq_pairwise_dist(x, y) ./ (2.0 * unwrap(k.α)))) .^ (-unwrap(k.α))
end
(k::RQ)(x::ArrayOrReal) = k(x, x)
Base.show(io::IO, k::RQ) = print(io, "RQ($(k.α))")

mutable struct SimilarHourKernel <: Kernel
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
function (k::SimilarHourKernel)(x::ArrayOrReal, y::ArrayOrReal)
    δ(x) = isapprox(x, 0.0, atol=1e-15) ? 1 : 0
    d = pairwise_dist(x, y) .% 24
    cs = unwrap(k.coeffs)
    hd = unwrap(k.hdeltas)
    K = cs[1] * δ.(d)
    if length(cs) > 1
        return K .+ sum(cs[i + 1] * (δ.(d .- (24 - i)) + δ.(d .- i)) for i in 1:(hd - 1))
    else
        return K
    end
end
(k::SimilarHourKernel)(x::ArrayOrReal) = k(x, x)
function Base.show(io::IO, k::SimilarHourKernel)
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
struct MA <: Kernel
    ν::Fixed
end
MA(n::Real) = MA(Fixed(n))
function (k::MA)(x::ArrayOrReal, y::ArrayOrReal)
    d = pairwise_dist(x, y)
    if unwrap(k.ν) ≈ 1/2
        return exp.(-d)
    elseif unwrap(k.ν) ≈ 3/2
        return (1 .+ √3 .* d) .* exp.(-√3 .* d)
    elseif unwrap(k.ν) ≈ 5/2
        return (1 .+ √5 .* d + 5/3 .* d.^2) .* exp.(-√5 .* d)
    else
        throw(ArgumentError("$(unwrap(k.ν)) is not a supported value for Matérn kernels."))
    end
end
(k::MA)(x::ArrayOrReal) = k(x, x)
Base.show(io::IO, k::MA) = print(io, "MA($(k.ν))")

"""
    RootLog <: Kernel

Kernel that computes (1/|x - y|) * log(1 + |x - y|).
"""
struct RootLog <: Kernel end
function (k::RootLog)(x::ArrayOrReal, y::ArrayOrReal)
    d = pairwise_dist(x, y)
    # The 1e-16 here is just to make sure that we get the correct limit when d → 0
    return (log.(d .+ 1) .+ 1e-16) ./ (d .+ 1e-16)
end
(k::RootLog)(x::ArrayOrReal) = k(x, x)
Base.show(io::IO, k::RootLog) = print(io, "RootLog()")

"""
    BinaryKernel <: Kernel

Kernel for binary inputs. Has three possible outcomes: Θ₁ if x = y = 1, Θ₂ if x = y = 0 and
Θ₃ if x ≠ y. Naturally, this only accepts unidimensional inputs.
"""
mutable struct BinaryKernel <: Kernel
    Θ₁
    Θ₂
    Θ₃
end
@unionise function (k::BinaryKernel)(x::AbstractArray{<:Integer}, y::AbstractArray{<:Integer})
    return unwrap(k.Θ₁) .* (x .≈ y' .≈ 1) .+
        unwrap(k.Θ₂) .* (x .≈ y' .≈ 0) .+
        unwrap(k.Θ₃) .* (x .!= y')
end
@unionise (k::BinaryKernel)(x::AbstractArray{<:Integer}) = k(x, x)
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
mutable struct ScaledKernel <: Kernel
    scale
    k::Kernel
end
isMulti(k::ScaledKernel) = isMulti(k.k)
function Base.:*(x, k::Kernel)
    return isconstrained(x) ?
        ScaledKernel(x, k) :
        (unwrap(x) ≈ zero(unwrap(x)) ? ZeroKernel() : ScaledKernel(Positive(x), k))
end
function Base.:*(x, k::ScaledKernel)
    if isconstrained(x)
        return ScaledKernel(x, k)
    else
        return unwrap(x) * unwrap(k.scale) ≈ zero(unwrap(x)) ? ZeroKernel() :
            (
                (isconstrained(k.scale) && !isa(k.scale, Positive)) ?
                ScaledKernel(Positive(x), k) :
                ScaledKernel(Positive(unwrap(x) * unwrap(k.scale)), k.k)
            )
    end
end
Base.:*(k::Kernel, x) = (*)(x, k::Kernel)
function Base.:+(k1::Kernel, k2::ScaledKernel)
    return unwrap(k2.scale) ≈ zero(unwrap(k2.scale)) ? k1 : SumKernel(k1, k2)
end
function Base.:+(k1::ScaledKernel, k2::Kernel)
    return unwrap(k1.scale) ≈ zero(unwrap(k1.scale)) ? k2 : SumKernel(k1, k2)
end
function Base.:+(k1::ScaledKernel, k2::ScaledKernel)
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
Base.show(io::IO, k::ScaledKernel) = print(io, "($(k.scale) * $(k.k))")

"""
    StretchedKernel <: Kernel

Represent any `Kernel` with length scale stretched to `stretch`.
"""
mutable struct StretchedKernel <: Kernel
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
    if length(lscale) > 1 && length(lscale) == size(x, 2) / 2
        return k.k(x ./ hcat(lscale, lscale), y ./ hcat(lscale, lscale))
    else
        return k.k(x ./ lscale, y ./ lscale)
    end
end
(k::StretchedKernel)(x) = k(x, x)
Base.show(io::IO, k::StretchedKernel) = print(io, "($(k.k) ▷ $(k.stretch))")

"""
    SumKernel <: Kernel

Kernel built by adding two kernels, `k1` and `k2`.
"""
mutable struct SumKernel <: Kernel
    k1::Kernel
    k2::Kernel
end
Base.:+(k1::Kernel, k2::Kernel) = SumKernel(k1, k2)
(k::SumKernel)(x, y) = k.k1(x, y) .+ k.k2(x, y)
(k::SumKernel)(x) = k(x, x)
Base.show(io::IO, k::SumKernel) = print(io, "($(k.k1) + $(k.k2))")
isMulti(k::SumKernel) = isMulti(k.k1) || isMulti(k.k2)

"""
    ProductKernel <: Kernel

Kernel built by multiplying two kernels, `k1` and `k2`.
"""
mutable struct ProductKernel <: Kernel
    k1::Kernel
    k2::Kernel
end
Base.:*(k1::Kernel, k2::Kernel) = ProductKernel(k1, k2)
function Base.:*(k1::Kernel, k2::ScaledKernel)
    return unwrap(k2.scale) ≈ zero(unwrap(k2.scale)) ? Kernel(0) : ProductKernel(k1, k2)
end
function Base.:*(k1::ScaledKernel, k2::Kernel)
    return unwrap(k1.scale) ≈ zero(unwrap(k1.scale)) ? Kernel(0) : ProductKernel(k1, k2)
end
function Base.:*(k1::ScaledKernel, k2::ScaledKernel)
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
Base.show(io::IO, k::ProductKernel) = print(io, "($(k.k1) * $(k.k2))")
isMulti(k::ProductKernel) = isMulti(k.k1) || isMulti(k.k2)

"""
    PeriodicKernel <: Kernel

Kernel built by defining a period `T` for kernel `k`.
"""
mutable struct PeriodicKernel <: Kernel
    T
    k::Kernel
end
function (k::PeriodicKernel)(x, y)
    px = [cos.(2π .* x ./ unwrap(k.T)') sin.(2π .* x ./ unwrap(k.T)')]
    py = [cos.(2π .* y ./ unwrap(k.T)') sin.(2π .* y ./ unwrap(k.T)')]
    return k.k(px, py)
end
(k::PeriodicKernel)(x) = k(x, x)
Base.show(io::IO, k::PeriodicKernel) = print(io, "($(k.k) ∿ $(k.T))")
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
mutable struct SpecifiedQuantityKernel <: Kernel
    col::Fixed
    k::Kernel
end
(←)(k::Kernel, s::Symbol) = SpecifiedQuantityKernel(Fixed(s), k)
function (k::SpecifiedQuantityKernel)(x::AbstractDataFrame, y::AbstractDataFrame)
    return k.k(disallowmissing(x[unwrap(k.col)]), disallowmissing(y[unwrap(k.col)]))
end
(k::SpecifiedQuantityKernel)(x::AbstractDataFrame) = k(x, x)
function (k::SpecifiedQuantityKernel)(x::DataFrameRow, y::DataFrameRow)
    return k(DataFrame(x), DataFrame(y))
end
function (k::SpecifiedQuantityKernel)(x::AbstractDataFrame, y::DataFrameRow)
    return k(x, DataFrame(y))
end
function (k::SpecifiedQuantityKernel)(x::DataFrameRow, y::AbstractDataFrame)
    return k(DataFrame(x), y)
end
Base.show(io::IO, k::SpecifiedQuantityKernel) = print(io, "($(k.k) ← $(k.col))")
isMulti(k::SpecifiedQuantityKernel) = isMulti(k.k)

"""
    ConstantKernel <: Kernel

Kernel that returns 1.0 for every pair of points.
"""
struct ConstantKernel <: Kernel end
(k::ConstantKernel)(x, y) = ones(Float64, size(x, 1), size(y, 1))
(k::ConstantKernel)(x) = k(x, x)
function Base.:+(k::Kernel, x)
    return isconstrained(x) ?
        SumKernel(k, x * ConstantKernel()) :
        (unwrap(x) ≈ zero(unwrap(x)) ? k : SumKernel(k, x * ConstantKernel()))
end
Base.:+(x, k::Kernel) = (+)(k::Kernel, x)
Base.convert(::Type{Kernel}, x::Real) = x ≈ 0.0 ? ZeroKernel() : Fixed(x) * ConstantKernel()
Base.show(io::IO, k::ConstantKernel) = print(io, "𝟏")

"""
    ZeroKernel <: Kernel

Zero kernel. Returns zero.
"""
struct ZeroKernel <: Kernel; end
(::ZeroKernel)(x, y) = zeros(size(x, 1), size(y, 1))
(k::ZeroKernel)(x) = k(x, x)
Base.:+(k::Kernel, z::ZeroKernel) = k
Base.:+(z::ZeroKernel, k::Kernel) = k + z
Base.:+(z::ZeroKernel, k::ZeroKernel) = z
Base.:+(z::ZeroKernel, k::ScaledKernel) = k
Base.:+(k::ScaledKernel, z::ZeroKernel) = z + k
Base.:*(k::Kernel, z::ZeroKernel) = z
Base.:*(z::ZeroKernel, k::Kernel) = k * z
Base.:*(z::ZeroKernel, k::ZeroKernel) = z
Base.:*(z::ZeroKernel, k::ScaledKernel) = z
Base.:*(k::ScaledKernel, z::ZeroKernel) = z * k
Base.:*(x, z::ZeroKernel) = z
Base.:*(z::ZeroKernel, x) = x * z
Base.show(io::IO, z::ZeroKernel) = print(io, "𝟎")

"""
    DiagonalKernel <: Kernel

Diagonal kernel. Has unitary variance.
"""
struct DiagonalKernel <: Kernel end
function (::DiagonalKernel)(x, y)
    xl = [x[i, :] for i in 1:size(x, 1)]
    yl = [y[i, :]' for i in 1:size(y, 1)]
    return float.(isapprox.(float.(xl), float.(yl')))
end
(k::GPForecasting.DiagonalKernel)(x::DataFrame, y::DataFrame) = k(Matrix(x), Matrix(y))
function (k::DiagonalKernel)(x::DataFrameRow, y::DataFrameRow)
    return k(DataFrame(x), DataFrame(y))
end
function (k::DiagonalKernel)(x::AbstractDataFrame, y::DataFrameRow)
    return k(x, DataFrame(y))
end
function (k::DiagonalKernel)(x::DataFrameRow, y::AbstractDataFrame)
    return k(DataFrame(x), y)
end
(k::DiagonalKernel)(x::Number, y) = k([x], y)
(k::DiagonalKernel)(x, y::Number) = k(x, [y])
(k::DiagonalKernel)(x::Number, y::Number) = k([x], [y])[1, 1]
(k::DiagonalKernel)(x) = k(x, x)
Base.show(io::IO, k::DiagonalKernel) = print(io, "δₓ")

"""
    DotKernel <: Kernel

Dot product kernel. Non-stationary.
"""
struct DotKernel <: Kernel end
@unionise function (::DotKernel)(x::AbstractArray{<:Real}, y::AbstractArray{<:Real})
    return x * y'
end
(k::DotKernel)(x::Number, y) = k([x], y)
(k::DotKernel)(x, y::Number) = k(x, [y])
(k::DotKernel)(x::Number, y::Number) = k([x], [y])
@unionise (k::DotKernel)(x::AbstractArray{<:Real}) = k(x, x)
Base.show(io::IO, k::DotKernel) = print(io, "<., .>")

"""
    HazardKernel <: Kernel

Kernel tailor-made for hazards. It uses an augmented version of the `DotKernel` and has two
fields, `bias` and `scale`. `bias` defaults to `Fixed(0.0)` and determines the projection of
every hazard vector into the non-hazard space. `scale` defaults to `Fixed(ones(d))`, where
`d` is the dimension of the hazards space and represents a (possibly non-uniform) scale to be
applied to every hazard vector, `v`, as `v .* scale`.

In order for this to make sense, `abs(bias)` has to be smaller than one and `scale` has to
be a `RowVector` when unwrapped. It probably makes more sense to use normalised hazard
vectors. Also, this is thought of as a multiplicative kernel.
"""
mutable struct HazardKernel <: Kernel
    bias
    scale

    function HazardKernel(b, s)
        unwrap(b) >= 1.0 && warn(
            """
            A HazardKernel with bias larger than or equal to 1.0 can yield unexpected
            results. Value received: $(unwrap(bias)).
            """
        )
        size(unwrap(s), 1) != 1 && throw(ArgumentError("Scale must be a `RowVector`"))
        return new(b, s)
    end
end
HazardKernel() = HazardKernel(Fixed(0.0), Fixed(-1))
HazardKernel(bias) = HazardKernel(bias, Fixed(-1))
@unionise function (k::HazardKernel)(x::AbstractArray{<:Real}, y::AbstractArray{<:Real})
    # First, augment the input space
    xl = [x[i, :] for i in 1:size(x, 1)]
    yl = [y[i, :] for i in 1:size(y, 1)]
    mask_x = isapprox.(xl, fill(zero(xl[1]), size(xl, 1))) # find which points have no hazard
    mask_y = isapprox.(yl, fill(zero(yl[1]), size(yl, 1)))
    h_x = .!mask_x .* fill(unwrap(k.bias), size(mask_x, 1)) # bias vector
    h_y = .!mask_y .* fill(unwrap(k.bias), size(mask_y, 1))
    scale = unwrap(k.scale) == -1 ? ones(1, size(x, 2)) : unwrap(k.scale)
    x_aug = hcat(x .* scale, (h_x + mask_x)) # scaling at the same time
    y_aug = hcat(y .* scale, (h_y + mask_y))

    return DotKernel()(x_aug, y_aug)
end
@unionise (k::HazardKernel)(x::AbstractArray{<:Real}) = k(x, x)
(k::HazardKernel)(x::Number, y) = k([x], y)
(k::HazardKernel)(x, y::Number) = k(x, [y])
(k::HazardKernel)(x::Number, y::Number) = k([x], [y])
Base.show(io::IO, k::HazardKernel) = print(io, "Hazard()")

"""
    ManifoldKernel <: Kernel

Build a kernel under the Manifold GPs framework. `k` is a regular kernel, while `NN` is a
neural network. The inputs to `k` are pre-transformed by `NN`. For more details, see:
"Manifold Gaussian Processes for Regression".
"""
struct ManifoldKernel <: Kernel
    k::Kernel
    NN::GPFNN
end
function (k::ManifoldKernel)(x, y)
    return k.k(
        vcat((k.NN(x[i, :])' for i in 1:size(x, 1))...),
        vcat((k.NN(y[i, :])' for i in 1:size(y, 1))...)
    )
end
function (k::ManifoldKernel)(x::T, y::P) where {T <: Input, P <: Input}
    return k.k(
        T(vcat((k.NN(x.val[i, :])' for i in 1:size(x.val, 1))...)),
        P(vcat((k.NN(y.val[i, :])' for i in 1:size(y.val, 1))...))
    )
end
(k::ManifoldKernel)(x) = k.k(vcat((k.NN(x[i, :])' for i in 1:size(x, 1))...))
function (k::ManifoldKernel)(x::T) where T <: Input
    return k.k(T(vcat((k.NN(x.val[i, :])' for i in 1:size(x.val, 1))...)))
end

Base.zero(::Kernel) = ZeroKernel()
Base.zero(::Type{GPForecasting.Kernel}) = ZeroKernel()

"""
    SparseKernel <: Kernel

Not supposed to be used directly by the user. This is automatically called under the hood,
whenever necessary.
"""
mutable struct SparseKernel <: Kernel
    k::Kernel
    Xm
    n
    σ²
end
# SparseKernel(k::Kernel, )
(k::SparseKernel)(x) = k.k(x, unwrap(k.Xm))
(k::SparseKernel)() = k.k(unwrap(k.Xm))
show(io::IO, k::SparseKernel) = print(io, "Sparse($(k.k))")
