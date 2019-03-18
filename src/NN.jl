# The reason why we are defining our own (very simple) NN architecture is that we want
# something that is Nabla-compatible (as we are going to do joint learning). Since we are not
# going to do anything remotely complex, this should be easy.
struct GPFNN <: AbstractNode
    layers::Vector{NNLayer}
end
function (nn::GPFNN)(x)
    layers = nn.layers
    out = x
    for layer in layers
        out = layer(out)
    end
    return out
end

mutable struct NNLayer <: AbstractNode
    W # weights
    B # biases
    σ::Fixed{Function} # activation function
end
(layer::NNLayer)(x) = layer.σ.(layer.W * x + layer.B)
