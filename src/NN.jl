# The reason why we are defining our own (very simple) NN architecture is that we want
# something that is Nabla-compatible (as we are going to do joint learning). Since we are not
# going to do anything remotely complex, this should be easy.
mutable struct NNLayer <: AbstractNode
    W # weights
    b # biases
    σ::Fixed{<:Function} # activation function
end
(layer::NNLayer)(x) = unwrap(layer.σ).(unwrap(layer.W) * x + unwrap(layer.b))
Base.size(l::NNLayer) = size(unwrap(l.W))
Base.size(l::NNLayer, i::Int) = size(unwrap(l.W), i)

mutable struct BatchNormLayer <: AbstractNode
    γ # if you want to use different values for each dimension, use a RowVector
    β # if you want to use different values for each dimension, use a RowVector
end
function (layer::BatchNormLayer)(x)
    means = mean(x, dims=1)
    stds = std(x, dims=1)
    x_norm = (x .- means) ./ (stds .+ 1e-10) # Adding 1e-10 for numerical stability
    return unwrap(layer.γ) .* x_norm .+ unwrap(layer.β)
end

struct GPFNN <: AbstractNode
    layers::Vector{Union{NNLayer, BatchNormLayer}}
end
function (nn::GPFNN)(x)
    layers = nn.layers
    out = x
    for layer in layers
        out = layer(out)
    end
    return out
end

# This type of layer is just really meant to be used by the neural kernel networks, so we'll
# only define it acting on kernels
mutable struct ProductLayer <: AbstractNode
    C::Fixed{Matrix{Bool}} # connectivity
end
(layer::ProductLayer)(x::Vector{<:Kernel}) = p_layer(unwrap(layer.C), x)
Base.size(l::ProductLayer) = size(unwrap(l.C))
Base.size(l::ProductLayer, i::Int) = size(unwrap(l.C), i)

# Just a few activation functions that may be handy
relu(x) = max(0, x)
noisy_relu(x; σ=0.01) = max(0, x + σ * randn())
leaky_relu(x; α=0.01) = x > 0.0 ? x : α * x
softplus(x) = log(1 + exp(x))
# Sigmoid "trick" from https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
sigmoid(x) = x < 0 ? exp(x)/(1 + exp(x)) : 1/(1 + exp(-x))
