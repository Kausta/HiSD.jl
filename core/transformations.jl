module Transformations

export Resize, RandomCrop, RandomHorizontalFlip, to_tensor, Normalize, ColorJitter, Compose

using Images, Interpolations
using Random
using Distributions

struct Resize
    size
    interpolation
    Resize(size, interpolation=BSpline(Linear())) = new(size, interpolation)
end
function (r::Resize)(x)
    if isa(r.size, Number)
        h, w = size(x)
        if (w <= h &&  w == r.size) || (h <= w && h == r.size)
            return x
        end

        oh, ow = if w < h
            Int(r.size * h / w), r.size
        else
           r.size, Int(r.size * w / h) 
        end
    else
        oh, ow = r.size
    end
    
    return imresize(x, (oh, ow), method=r.interpolation)
end

struct RandomCrop
    size
    RandomCrop(size::Number) = new((size, size))
    RandomCrop(size) = new(size)
end
function get_params(r::RandomCrop, x)
    h, w = size(x)
    th, tw = r.size
    if h == th && w == tw
        return 1, 1, h, w
    end

    i = rand(1:(h-th+1))
    j = rand(1:(w-tw+1))
    return i, j, th, tw
end
function (r::RandomCrop)(x)
    i, j, h, w = get_params(r, x)
    return x[i:i+h-1, j:j+w-1]
end

struct RandomHorizontalFlip
    prob
    RandomHorizontalFlip(prob=0.5) = new(prob)
end
function (r::RandomHorizontalFlip)(x)
    if rand() < r.prob
        x = x[:, end:-1:1]
    end
    return x
end

function to_tensor(x)
    return x |> channelview
end

struct Normalize
    mean
    std
    Normalize(mean, std) = new(mean, std)
end
(n::Normalize)(x) = (x .- n.mean) ./ n.std

function blend(im1, im2, alpha)
    return clamp.((1 - alpha) * im1 .+ alpha * im2, 0.0, 1.0)
end

function gray(img)
   return 0.299 * img[1, :, :] + 0.587 * img[2, :, :] + 0.114 * img[3, :, :]
end

function adjust_brightness(img, brightness_factor)
    return clamp.(brightness_factor * img, 0.0, 1.0)
end

function adjust_contrast(img, contrast_factor)
    degenerate = mean(gray(img))
    return blend(degenerate, img, contrast_factor)
end

function adjust_saturation(img, saturation_factor)
    degenerate = gray(img)
    return blend(reshape(degenerate, (1,size(degenerate)...)), img, saturation_factor)
end

function adjust_hue(img, hue_factor)
    img = img |> colorview(RGB) .|> HSV |> channelview
    scale = 360.0
    img[1, :, :] = mod.(img[1, :, :] .+ scale * hue_factor, scale)
    return img |> colorview(HSV) .|> RGB |> channelview
end

function get_jitter_param(value, name, center=1, clip_first_on_zero=true)
    @assert value >= 0 "$name must be non-negative"
    value = [center - value, center + value]
    if clip_first_on_zero
        value[1] = max(value[1], 0)
    end

    # if value is 0 or (1., 1.) for brightness/contrast/saturation
    # or (0., 0.) for hue, do nothing
    if value[1] == center && value[2] == center
        value = nothing
    end
    return value
end

struct ColorJitter
    brightness
    contrast
    saturation
    hue
    function ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)
        brightness = get_jitter_param(brightness, "brightness")
        contrast = get_jitter_param(contrast, "contrast")
        saturation = get_jitter_param(saturation, "saturation")
        hue = get_jitter_param(hue, "hue", 0, false)
        new(brightness, contrast, saturation, hue)
    end
end
function (c::ColorJitter)(x)
    transforms = []
    if !isnothing(c.brightness)
        brightness_factor = rand(Uniform(c.brightness[1], c.brightness[2]))
        push!(transforms, img -> adjust_brightness(img, brightness_factor)) 
    end
    if !isnothing(c.contrast)
        contrast_factor = rand(Uniform(c.contrast[1], c.contrast[2]))
        push!(transforms, img -> adjust_contrast(img, contrast_factor)) 
    end
    if !isnothing(c.saturation)
        saturation_factor = rand(Uniform(c.saturation[1], c.saturation[2]))
        push!(transforms, img -> adjust_saturation(img, saturation_factor)) 
    end
    if !isnothing(c.hue)
        hue_factor = rand(Uniform(c.hue[1], c.hue[2]))
        push!(transforms, img -> adjust_hue(img, hue_factor)) 
    end
    shuffle!(transforms)
    img = x |> channelview
    img = Compose(transforms...)(img)
    return img |> colorview(RGB)
end

struct Compose
    transformations
    Compose(transformations...) = new(transformations)
end
(c::Compose)(x) = (for t in c.transformations; x = t(x); end; x)

end