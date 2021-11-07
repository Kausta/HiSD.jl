module Network

using Knet
include("primitives.jl")
using .Primitives

export DownBlock
struct DownBlock; conv1; conv2; sc; activ; end
DownBlock(in_ch::Int,out_ch::Int) = 
    DownBlock(Conv(in_ch, in_ch, 3, 1), Conv(in_ch, out_ch, 3, 1), Conv(in_ch, out_ch, 1, 0, bias=false), LeakyRelu(0.2))
function (d::DownBlock)(x)
    residual = avg_pool2d(d.sc(x), 2)
    out = d.conv2(d.activ(avg_pool2d(d.conv1(d.activ(x)), 2)))
    return (residual + out) / sqrt(2)  
end

export DownBlockIN
struct DownBlockIN; conv1; conv2; in1; in2; sc; activ; end
DownBlockIN(in_ch::Int,out_ch::Int) = 
    DownBlockIN(Conv(in_ch, in_ch, 3, 1), Conv(in_ch, out_ch, 3, 1), 
                InstanceNorm2d(in_ch), InstanceNorm2d(in_ch),
                Conv(in_ch, out_ch, 1, 0, bias=false), LeakyRelu(0.2))
function (d::DownBlockIN)(x)
    residual = avg_pool2d(d.sc(x), 2)
    out = d.conv2(d.activ(d.in2(avg_pool2d(d.conv1(d.activ(d.in1(x))), 2))))
    return (residual + out) / sqrt(2)  
end

export UpBlock
struct UpBlock; conv1; conv2; sc; activ; end
UpBlock(in_ch::Int,out_ch::Int) = 
    UpBlock(Conv(in_ch, out_ch, 3, 1), Conv(out_ch, out_ch, 3, 1), Conv(in_ch, out_ch, 1, 0, bias=false), LeakyRelu(0.2))
function (d::UpBlock)(x)
    residual = upsample2d(d.sc(x), 2)
    out = d.conv2(d.activ(d.conv1(upsample2d(d.activ(x), 2))))
    return (residual + out) / sqrt(2)  
end

export UpBlockIn
struct UpBlockIn; conv1; conv2; in1; in2; sc; activ; end
UpBlockIn(in_ch::Int,out_ch::Int) = 
    UpBlockIn(Conv(in_ch, out_ch, 3, 1), Conv(out_ch, out_ch, 3, 1), 
              InstanceNorm2d(in_ch), InstanceNorm2d(out_ch),
              Conv(in_ch, out_ch, 1, 0, bias=false), LeakyRelu(0.2))
function (d::UpBlockIn)(x)
    residual = upsample2d(d.sc(x), 2)
    out = d.conv2(d.activ(d.in2(d.conv1(upsample2d(d.activ(d.in1(x)), 2)))))
    return (residual + out) / sqrt(2)  
end


export MiddleBlock, num_adain_params, assign_adain_params
struct MiddleBlock; conv1; conv2; adain1; adain2; sc; activ; end
MiddleBlock(in_ch::Int,out_ch::Int) = 
    MiddleBlock(Conv(in_ch, out_ch, 3, 1), Conv(out_ch, out_ch, 3, 1), 
                AdaptiveInstanceNorm2d(in_ch), AdaptiveInstanceNorm2d(out_ch),
                Conv(in_ch, out_ch, 1, 0, bias=false), LeakyRelu(0.2))
function (d::MiddleBlock)(x)
    residual = d.sc(x)
    out = d.conv2(d.activ(d.adain2(d.conv1(d.activ(d.adain1(x))))))
    return (residual + out) / sqrt(2)  
end
Primitives.num_adain_params(d::MiddleBlock) = num_adain_params(d.adain1) + num_adain_params(d.adain2)
function Primitives.assign_adain_params(d::MiddleBlock, params)
    params = assign_adain_params(d.adain1, params)
    params = assign_adain_params(d.adain2, params)
    return params
end

#TODO: Extractor, Translator and Mapper
#TODO: Generator, Discriminator

end