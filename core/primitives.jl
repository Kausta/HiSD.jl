module Primitives

using Knet
using Statistics: mean, std

export kaiming_normal
function kaiming_normal(a...) # mode="fan_in", nonlinearity="relu"
    # implementation based on pytorch 1.0.1 kaiming normal
    # same as the original paper
    # only for mode="fan_in", nonlinearity="relu"
    # since others are not needed
    w = randn(a...)
    
    if ndims(w) == 1
        fan_in = length(w)
    elseif ndims(w) == 2
        fan_in = size(w,2)
    else
        # if a is (3,3,16,8), then there are 16 input channels and 8 output channels
        # fanin = 3*3*16 = (3*3*16*8) ÷ 8
        # fanout = 3*3*8 = (3*3*16*8) ÷ 16
        fan_in = div(length(w),  a[end])
    end
    fan = fan_in
    gain = sqrt(2)
    std = convert(eltype(w), gain / sqrt(fan))
    return w .* std
end

export Chain
struct Chain
    layers
    Chain(layers...) = new(layers)
end
(c::Chain)(x) = (for l in c.layers; x = l(x); end; x)

export Conv
struct Conv; w; b; p; s; end
function Conv(in_ch::Int,out_ch::Int,ks::Int,padding::Int=0,stride::Int=1;bias::Bool=true) 
    w = param(ks,ks,in_ch,out_ch,init=kaiming_normal)
    b = nothing
    if bias
        b = param0(1,1,out_ch,1)
    end
    Conv(w, b, padding, stride)
end
function (c::Conv)(x) 
    res = conv4(c.w, x, padding=c.p, stride=c.s) 
    if !isnothing(c.b)
        res = res .+ c.b
    end
    res
end

export avg_pool2d, AvgPool2d
avg_pool2d(x,ks) = pool(x, window=ks, stride=ks, padding=0, mode=1)
struct AvgPool2d; ks; end
(a::AvgPool2d)(x) = avg_pool2d(x, a.ks)

export adaptive_avg_pool2d, AdaptiveAvgPool2d
# NOTE: Only equivalent to PyTorch AdaptiveAvgPool2d when output size is an integer multiple of the input sizes
#       When output size is not an integer multiple of the input size, PyTorch creates kernels of variable size
#       We instead produce a static kernel size equivalent to PyTorch's implementation 
#          when output size is an integer multiple of the input sizes
#       Since the model only requires adaptive pooling with output size = 1, this is not a problem for HiSD
function adaptive_avg_pool2d(x,os) 
    ind = size(x)[1:2]
    stride = ind .÷ os
    kernel_size = ind .- (os.-1) .* stride
    pool(x, window=kernel_size, stride=stride, padding=0, mode=1)
end
struct AdaptiveAvgPool2d; os; end;
(a::AdaptiveAvgPool2d)(x) = adaptive_avg_pool2d(x, a.os)

export upsample2d, Upsample2d
# Nearest neighbor upsampling
upsample2d(x,sf) = unpool(x, window=sf, stride=sf, padding=0, mode=1)
struct Upsample2d; sf; end
(u::Upsample2d)(x) = upsample2d(x, u.sf)

export leaky_relu, LeakyRelu
leaky_relu(x, alpha=0.2) = max.(0,x) .+ (min.(0,x) .* eltype(x)(alpha))
struct LeakyRelu; alpha; end
(l::LeakyRelu)(x) = leaky_relu(x, l.alpha)

export InstanceNorm2d
struct InstanceNorm2d; weight; bias; num_features::Int; eps; end
InstanceNorm2d(num_features; eps=1e-5) = InstanceNorm2d(param(ones(1, num_features, 1)), param0(1, num_features, 1), num_features, eps)
function (i::InstanceNorm2d)(x)
    h, w, c, n = size(x)
    x = reshape(x, (h * w, c, n))
    bias_in = mean(x, dims=1)
    weight_in = std(x, mean=bias_in, dims=1)
    
    out = ((x .- bias_in) ./ (weight_in .+ i.eps)) .* i.weight .+ i.bias
    return reshape(out, (h, w, c, n))
end


export AdaptiveInstanceNorm2d, num_adain_params, assign_adain_params
mutable struct AdaptiveInstanceNorm2d; weight; bias; num_features::Int; eps; end
AdaptiveInstanceNorm2d(num_features; eps=1e-5) = AdaptiveInstanceNorm2d(nothing, nothing, num_features, eps)
num_adain_params(i::AdaptiveInstanceNorm2d) = 2 * i.num_features
function assign_adain_params(i::AdaptiveInstanceNorm2d, params)
    nf = i.num_features
    b = size(params)[2]
    i.bias = reshape(params[1:nf, :], (1, nf, b))
    i.weight = reshape(params[nf+1:2*nf,:], (1, nf, b)) .+ 1
    if size(params)[2] > 2 * nf
        return params[:, 2*nf+1:end]
    else
        return params
    end
end
function (i::AdaptiveInstanceNorm2d)(x)
    h, w, c, n = size(x)
    x = reshape(x, (h * w, c, n))
    bias_in = mean(x, dims=1)
    weight_in = std(x, mean=bias_in, dims=1)
    
    out = ((x .- bias_in) ./ (weight_in .+ i.eps)) .* i.weight .+ i.bias
    return reshape(out, (h, w, c, n))
end

export Linear
struct Linear; w; b; end
Linear(in_dim::Int, out_dim::Int) = Linear(param(out_ch,in_ch,init=kaiming_normal), param0(out_ch))
(l::Linear)(x) = l.w * relu.(x) .+ l.b

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

end