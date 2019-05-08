# The reason why we are defining our own (very simple) NN architecture is that we want
# something that is Nabla-compatible (as we are going to do joint learning). Since we are not
# going to do anything remotely complex, this should be easy.
mutable struct NNLayer <: AbstractNode
    W # weights
    b # biases
    σ::Fixed{<:Function} # activation function
end
(layer::NNLayer)(x) = unwrap(layer.σ).(unwrap(layer.W) * x + unwrap(layer.b))

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

# Just a few activation functions that may be handy
relu(x) = max(0, x)
noisy_relu(x; σ=0.01) = max(0, x + σ * randn())
leaky_relu(x; α=0.01) = x > 0.0 ? x : α * x
softplus(x) = log(1 + exp(x))
sigmoid(x) = 1/(1 + exp(-x))
