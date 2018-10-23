# Functions

## Kernel functions

```@docs
Kernel
EQ
RQ
MA
BinaryKernel
ConstantKernel
DiagonalKernel
ZeroKernel
ScaledKernel
StretchedKernel
stretch
SumKernel
PeriodicKernel
periodicise
SpecifiedQuantityKernel
PosteriorKernel
NoiseKernel
MultiKernel
LMMKernel
verynaiveLMMKernel
NaiveLMMKernel
LMMPosKernel
isMulti
hourly_cov
```

## Mean functions

```@docs
Mean
ConstantMean
FunctionMean
ZeroMean
PosteriorMean
ScaledMean
MultiMean
LMMPosMean
```

## GP Functions

```@docs
GP
condition
credible_interval
```

## Parameter Functions

```@docs
Parameter
Fixed
Positive
Named
Bounded
DynamicBound
isconstrained
```

## Other Functions

```@docs
logpdf
objective
minimise
learn
pairwise_dist
sq_pairwise_dist
Input
Observed
Latent
```
