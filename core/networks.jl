using Statistics: mean
using Base.Iterators
using AutoGrad

sq2 = Float32(sqrt(Float32(2.)))

export DownBlock
struct DownBlock; conv1; conv2; sc; activ; end
DownBlock(in_ch::Int,out_ch::Int) = 
    DownBlock(Conv(in_ch, in_ch, 3, 1), Conv(in_ch, out_ch, 3, 1), Conv(in_ch, out_ch, 1, 0, bias=false), LeakyRelu(0.2))
function (d::DownBlock)(x)
    residual = avg_pool2d(d.sc(x), 2)
    out = d.conv2(d.activ(avg_pool2d(d.conv1(d.activ(x)), 2)))
    return (residual + out) / sq2
end
paramlist(d::DownBlock) = Iterators.flatten(paramlist.([d.conv1, d.conv2, d.sc]))

export DownBlockIN
struct DownBlockIN; conv1; conv2; in1; in2; sc; activ; end
DownBlockIN(in_ch::Int,out_ch::Int) = 
    DownBlockIN(Conv(in_ch, in_ch, 3, 1), Conv(in_ch, out_ch, 3, 1), 
                InstanceNorm2d(in_ch), InstanceNorm2d(in_ch),
                Conv(in_ch, out_ch, 1, 0, bias=false), LeakyRelu(0.2))
function (d::DownBlockIN)(x)
    residual = avg_pool2d(d.sc(x), 2)
    out = d.conv2(d.activ(d.in2(avg_pool2d(d.conv1(d.activ(d.in1(x))), 2))))
    return (residual + out) / sq2
end
paramlist(d::DownBlockIN) = Iterators.flatten(paramlist.([d.conv1, d.conv2, d.in1, d.in2, d.sc]))

export UpBlock
struct UpBlock; conv1; conv2; sc; activ; end
UpBlock(in_ch::Int,out_ch::Int) = 
    UpBlock(Conv(in_ch, out_ch, 3, 1), Conv(out_ch, out_ch, 3, 1), Conv(in_ch, out_ch, 1, 0, bias=false), LeakyRelu(0.2))
function (d::UpBlock)(x)
    residual = upsample2d(d.sc(x), 2)
    out = d.conv2(d.activ(d.conv1(upsample2d(d.activ(x), 2))))
    return (residual + out) / sq2
end
paramlist(d::UpBlock) = Iterators.flatten(paramlist.([d.conv1, d.conv2, d.sc]))

export UpBlockIN
struct UpBlockIN; conv1; conv2; in1; in2; sc; activ; end
UpBlockIN(in_ch::Int,out_ch::Int) = 
    UpBlockIN(Conv(in_ch, out_ch, 3, 1), Conv(out_ch, out_ch, 3, 1), 
              InstanceNorm2d(in_ch), InstanceNorm2d(out_ch),
              Conv(in_ch, out_ch, 1, 0, bias=false), LeakyRelu(0.2))
function (d::UpBlockIN)(x)
    residual = upsample2d(d.sc(x), 2)
    out = d.conv2(d.activ(d.in2(d.conv1(upsample2d(d.activ(d.in1(x)), 2)))))
    return (residual + out) / sq2
end
paramlist(d::UpBlockIN) = Iterators.flatten(paramlist.([d.conv1, d.conv2, d.in1, d.in2, d.sc]))

export MiddleBlock, num_adain_params, assign_adain_params
struct MiddleBlock; conv1; conv2; adain1; adain2; sc; activ; end
MiddleBlock(in_ch::Int,out_ch::Int) = 
    MiddleBlock(Conv(in_ch, out_ch, 3, 1), Conv(out_ch, out_ch, 3, 1), 
                AdaptiveInstanceNorm2d(in_ch), AdaptiveInstanceNorm2d(out_ch),
                Conv(in_ch, out_ch, 1, 0, bias=false), LeakyRelu(0.2))
function (d::MiddleBlock)(x)
    residual = d.sc(x)
    out = d.conv2(d.activ(d.adain2(d.conv1(d.activ(d.adain1(x))))))
    return (residual + out) / sq2
end
num_adain_params(d::MiddleBlock) = num_adain_params(d.adain1) + num_adain_params(d.adain2)
function assign_adain_params(d::MiddleBlock, params)
    params = assign_adain_params(d.adain1, params)
    params = assign_adain_params(d.adain2, params)
    return params
end
paramlist(d::MiddleBlock) = Iterators.flatten(paramlist.([d.conv1, d.conv2, d.sc]))

export LinearBlock
struct LinearBlock; l; end
LinearBlock(in_dim::Int, out_dim::Int) = LinearBlock(Linear(in_dim, out_dim))
(l::LinearBlock)(x) = l.l(relu.(x))
paramlist(d::LinearBlock) = paramlist(d.l)

export Extractors
struct Extractors; num_tags; model; end
function Extractors(config) 
    num_tags = length(config["tags"])
    channels = config["extractors"]["channels"]
    Extractors(num_tags, Chain(
            Conv(config["input_dim"], channels[1], 1, 0), 
            [DownBlock(channels[i], channels[i + 1]) for i in 1:(length(channels)-1)]...,
            AdaptiveAvgPool2d(1),
            Conv(channels[end],  config["style_dim"] * num_tags, 1, 0),
        )
    )
end
function (e::Extractors)(x, i)
    sty = e.model(x)
    sz = size(sty)
    sty = reshape(sty, (prod(sz) ÷ (e.num_tags * sz[end]), e.num_tags, sz[end]))
    return sty[:, i, :]
end
paramlist(d::Extractors) = paramlist(d.model)

export Mapper
struct Mapper; pre_model; post_models; end
function Mapper(config, num_attributes::Int64)
    channels = config["mappers"]["pre_channels"]
    pre_model = Chain(
        Linear(config["noise_dim"], channels[1]),
        [LinearBlock(channels[i], channels[i+1]) for i in 1:(length(channels)-1)]...
    )
    channels = config["mappers"]["post_channels"]
    post_models = [
        Chain(
            [LinearBlock(channels[i], channels[i+1]) for i in 1:(length(channels)-1)]...,
            Linear(channels[end], config["style_dim"])
        ) for i in 1:num_attributes
    ]
    Mapper(pre_model, post_models)
end
(m::Mapper)(z, j) = m.post_models[j](m.pre_model(z))
paramlist(d::Mapper) = Iterators.flatten((paramlist(d.pre_model), Iterators.flatten(paramlist.(d.post_models))))

export Translator
struct Translator; model; style_to_params; features; masks; end
function Translator(config)
    channels = config["translators"]["channels"]
    model = Chain(
        Conv(config["encoder"]["channels"][end], channels[1], 1, 0),
        [MiddleBlock(channels[i], channels[i+1]) for i in 1:(length(channels)-1)]...
    )
    style_to_params = Linear(config["style_dim"], num_adain_params(model))
    features = Chain(
        Conv(channels[end], config["decoder"]["channels"][1], 1, 0)
    )
    masks = Chain(
        Conv(channels[end], config["decoder"]["channels"][1], 1, 0),
        Sigmoid()
    )
    Translator(model, style_to_params, features, masks)
end
function (t::Translator)(e, s)
    p = t.style_to_params(s)
    assign_adain_params(t.model, p)
    
    mid = t.model(e)
    f = t.features(mid)
    m = t.masks(mid)
    
    return f .* m + e .* (1 .- m)
end
paramlist(d::Translator) = Iterators.flatten(paramlist.([d.model, d.style_to_params, d.features, d.masks]))

export Generator, encode, decode, extract, map, translate
struct Generator; tags; style_dim; noise_dim; encoder; decoder; extractors; translators; mappers; end
function Generator(config)
    tags = config["tags"]
    style_dim = config["style_dim"]
    noise_dim = config["noise_dim"]
    
    channels = config["encoder"]["channels"]
    encoder = Chain(
        Conv(config["input_dim"], channels[1], 1, 0),
        [DownBlockIN(channels[i], channels[i+1]) for i in 1:(length(channels)-1)]...
    )
    
    channels = config["decoder"]["channels"]
    decoder = Chain(
        [UpBlockIN(channels[i], channels[i+1]) for i in 1:(length(channels)-1)]...,
        Conv(channels[end], config["input_dim"], 1, 0)
    )
    
    extractors = Extractors(config)
    translators = [Translator(config) for i in 1:length(tags)]
    mappers = [Mapper(config, length(tags[i]["attributes"])) for i in 1:length(tags)]
    
    Generator(tags, style_dim, noise_dim, encoder, decoder, extractors, translators, mappers)
end
encode(g::Generator, x) = g.encoder(x)
decode(g::Generator, e) = g.decoder(e)
extract(g::Generator, x, i) = g.extractors(x, i)
map(g::Generator, z, i, j) = g.mappers[i](z, j)
translate(g::Generator, e, s, i) = g.translators[i](e, s)

paramlist(g::Generator) = Iterators.flatten((paramlist_others(g), paramlist_mappers(g)))
function paramlist_others(d::Generator) 
    f = Iterators.flatten(paramlist.([d.encoder, d.decoder, d.extractors]))
    t = Iterators.flatten(paramlist.(d.translators))
    return Iterators.flatten((f, t))
end
paramlist_mappers(d::Generator) = Iterators.flatten(paramlist.(d.mappers))

export Discriminator
struct Discriminator; tags; conv; fcs; end
function Discriminator(config)
    tags = config["tags"]
    channels = config["discriminators"]["channels"]
    conv = Chain(
        Conv(config["input_dim"], channels[1], 1, 0),
        [DownBlock(channels[i], channels[i+1]) for i in 1:(length(channels)-1)]...,
        AdaptiveAvgPool2d(1)
    )
    
    fcs = [
        Chain(
            Conv(channels[end] + 
                 # ALI part, not shown in original submission
                 # config["style_dim"] + 
                 # Tag-irrelevant part. Sec.3.4
                 tags[i]["tag_irrelevant_conditions_dim"],
                 # One for translated, one for cycle. Eq.4
                 length(tags[i]["attributes"]) * 2, 1, 0)
        ) for i in 1:length(tags)
    ]
    
    Discriminator(tags, conv, fcs)
end
function (d::Discriminator)(x, #=s,=# y, i)
    f = d.conv(x)
    
    B = size(f)[end]
    # s = reshape(s, (1, 1, prod(size(s)) ÷ B, B))
    f = reshape(f, (prod(size(f)) ÷ B, B))
    y = reshape(y, (prod(size(y)) ÷ B, B))
    fsy = vcat(f, #= s, =# y)
    fsy = reshape(fsy, (1, 1, size(fsy)[1], B))
    out = d.fcs[i](fsy)
    so = size(out)
    return reshape(out, (prod(so) ÷ (2*so[end]), 2, so[end]))
end
paramlist(d::Discriminator) = Iterators.flatten(paramlist.([d.conv, d.fcs]))

export dis_loss_real, dis_loss_fake_trg, dis_loss_fake_cyc, gen_loss_real, gen_loss_fake_trg, gen_loss_fake_cyc
function dis_loss_real(d::Discriminator, x, y, i, j)
    out = d(x, y, i)[j, :, :]
    loss = mean(relu.(1 .- out[1, :])) + mean(relu.(1 .- out[2, :]))
    # R1 regularization 
    # Expected functionality: loss += grad(out[1, :], x) + grad(out[1, :], x)
    # Below is commented as it does not work in this form (or without @diffs)
    # due to cat/uncat
    xp = Param(x)
    gout1 = @diff sum(d(xp, y, i)[j, 1, :])
    gout1_grad = grad(gout1, xp)
    loss += sum(abs2.(gout1_grad)) / size(x)[end]
    gout2 = @diff sum(d(xp, y, i)[j, 2, :])
    gout2_grad = grad(gout2, xp)
    loss += sum(abs2.(gout2_grad)) / size(x)[end]
    return loss
end
function dis_loss_fake_trg(d::Discriminator, x, y, i, j)
    out = d(x, y, i)[j, :, :]
    loss = mean(relu.(1 .+ out[1, :]))
    return loss
end
function dis_loss_fake_cyc(d::Discriminator, x, y, i, j)
    out = d(x, y, i)[j, :, :]
    loss = mean(relu.(1 .+ out[2, :]))
    return loss
end
function gen_loss_real(d::Discriminator, x, y, i, j)
    out = d(x, y, i)[j, :, :]
    loss = mean(out[1, :]) + mean(out[2, :])
    return loss
end
function gen_loss_fake_trg(d::Discriminator, x, y, i, j)
    out = d(x, y, i)[j, :, :]
    loss = -mean(out[1, :])
    return loss
end
function gen_loss_fake_cyc(d::Discriminator, x, y, i, j)
    out = d(x, y, i)[j, :, :]
    loss = -mean(out[2, :])
    return loss
end