using GPForecasting
using StatsBase
using Statistics
using LinearAlgebra
using Random
using Plots

# sample some data from a GOLMM
p, m = 4, 2;
ks = [stretch(EQ(), 1.0), stretch(EQ(), 7.0)];
group_embs = [1.0, 4.5, 1.1, 4.6];
group_k = EQ();

k_golmm = GOLMMKernel(m, p, 0.001, 0.001, ks, group_k, group_embs);
gp = GP(k_golmm);

ts = collect(0.0:0.05:20.0);
ys = sample(gp(ts));

# create a new GOLMM and fit it to data
k_olmm_init = GOLMMKernel(m, p, Fixed(0.001), Fixed(0.001), ks, EQ(), [1., 2., 3., 4.]);
gp = GP(k_olmm_init);
gp = learn(gp, ts, ys, mle_obj, its=50);

# perform inference
posterior_gp = condition(gp, ts[1:100], ys[1:100, :]);
plot(sample(posterior_gp(ts)), label="sample")
plot!(ys, label="train", linestyle=:dash)

k = EQ()

activation = sigmoid
l1 = NNLayer(randn(6, 1), 1e-2 .* randn(6), Fixed(activation))
l2 = NNLayer(randn(2, 6), 1e-2 .* randn(2), Fixed(activation))
nn = GPFNN([l1, l2])
mk = ManifoldKernel(k, nn)

x = rand(50)
 GPForecasting.is_not_noisy(mk)
 isa(mk(x), AbstractMatrix)
 mk(x) ≈ mk(x, x)
 diag(mk(x)) ≈ var(mk, x)
hourly_cov(mk, x) ≈ Diagonal(var(mk, x))

k = NoiseKernel(1.0 * stretch(EQ(), Positive([1.0, 1.0])), Fixed(1e-2) * DiagonalKernel())
x = Observed(x)
mk = ManifoldKernel(k, nn)
!GPForecasting.is_not_noisy(mk)
i*sa(mk(x), AbstractMatrix)
mk(x) ≈ mk(x, x) atol = _ATOL_ rtol = _RTOL_
diag(mk(x)) ≈ var(mk, x) atol = _ATOL_ rtol = _RTOL_
hourly_cov(mk, x) ≈ Diagonal(var(mk, x)) atol = _ATOL_ rtol = _RTOL_
