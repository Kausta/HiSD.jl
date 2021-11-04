module Network

using Knet
using AutoGrad: AutoGrad, @primitive

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
avg_pool2d(x,ks::Int) = pool(x, window=ks, stride=ks, padding=0, mode=1)
struct AvgPool2d; ks::Int; end
(a::AvgPool2d)(x) = avg_pool2d(x, a.ks)
#TODO: AdaptiveAvgPool2d

export leaky_relu, LeakyRelu
#TODO: @primitive
leaky_relu(x, alpha=0.2) = max.(0,x) .+ (min.(0,x) .* eltype(x)(alpha))
struct LeakyRelu; alpha; end
(l::LeakyRelu)(x) = leaky_relu(x, l.alpha)

export DownBlock
struct DownBlock; conv1; conv2; sc; activ; end
DownBlock(in_ch::Int,out_ch::Int) = 
    DownBlock(Conv(in_ch, in_ch, 3, 1), Conv(in_ch, out_ch, 3, 1), Conv(in_ch, out_ch, 1, 0, bias=false), LeakyRelu(0.2))
function (d::DownBlock)(x)
    residual = avg_pool2d(d.sc(x), 2)
    out = d.conv2(d.activ(avg_pool2d(d.conv1(d.activ(x)), 2)))
    return (residual + out) / sqrt(2)  
end

#TODO: DownBlockIN

export upsample2d, Upsample2d
#TODO: NNLib upsample_nearest, ∇upsample_nearest with @primitive1 for 2d nearest interpolation
# unpool is temporary
upsample2d(x,sf::Int) = unpool(x, window=sf, stride=sf, padding=0, mode=1)
struct Upsample2d; sf::Int; end
(u::Upsample2d)(x) = upsample2d(x, u.sf)

export UpBlock
struct UpBlock; conv1; conv2; sc; activ; end
UpBlock(in_ch::Int,out_ch::Int) = 
    UpBlock(Conv(in_ch, out_ch, 3, 1), Conv(out_ch, out_ch, 3, 1), Conv(in_ch, out_ch, 1, 0, bias=false), LeakyRelu(0.2))
function (d::UpBlock)(x)
    residual = upsample2d(d.sc(x), 2)
    out = d.conv2(d.activ(d.conv1(upsample2d(d.activ(x), 2))))
    return (residual + out) / sqrt(2)  
end

#TODO: UpBlockIN

export MiddleBlock
#TODO: AdaptiveInstanceNorm2d for Middle Block
struct MiddleBlock; conv1; conv2; sc; activ; end
MiddleBlock(in_ch::Int,out_ch::Int) = 
    MiddleBlock(Conv(in_ch, out_ch, 3, 1), Conv(out_ch, out_ch, 3, 1), Conv(in_ch, out_ch, 1, 0, bias=false), LeakyRelu(0.2))
function (d::MiddleBlock)(x)
    residual = d.sc(x)
    out = d.conv2(d.activ(d.conv1(d.activ(x))))
    return (residual + out) / sqrt(2)  
end

export Linear
struct Linear; w; b; end
Linear(in_dim::Int, out_dim::Int) = Linear(param(out_ch,in_ch,init=kaiming_normal), param0(out_ch))
(l::Linear)(x) = l.w * relu.(x) .+ l.b

#TODO: Extractor, Translator and Mapper
#TODO: Generator, Discriminator

end