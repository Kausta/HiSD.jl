module Data

using Images
using Images.FileIO
import Base: iterate, eltype, length, rand, repeat, summary, show
using Base: @propagate_inbounds, tail
using Random: randperm

export ImageData, ImageDataset, ImageDatasetBatch, minibatch

struct ImageData; filename::String; gender::Int; age::Int; end

function load(d::ImageData, transformation=nothing)
    image = FileIO.load(d.filename)
    conditions = [d.gender, d.age]
    if !isnothing(transformation)
        image = transformation(image)
    end
    return image, conditions
end

struct ImageDataset
    lines::Vector{ImageData}
    transformation

    function ImageDataset(root_dir::AbstractString, filename::AbstractString, transformation=nothing)
        path = joinpath(root_dir, filename)
        lines = readlines(path)
        lines = (rstrip(line) |> split for line in lines)
        new([ImageData(line[1], parse(Int, line[2]), parse(Int, line[3])) for line in lines], transformation)
    end
end

load(d::ImageDataset, idx) = load(d.lines[idx], d.transformation)
Base.length(d::ImageDataset) = length(d.lines)
Base.iterate(d::ImageDataset, state=1) = state > length(d.lines) ? nothing : (load(d, state), state + 1)

mutable struct ImageDatasetBatch; image_dataset::ImageDataset; batchsize; length; partial; imax; indices; shuffle; xtype; ytype; end
 
function minibatch(image_dataset::ImageDataset, batchsize, atype; shuffle=false, partial=true)
    nx = length(image_dataset)
    imax = partial ? nx : nx - batchsize + 1
    ImageDatasetBatch(image_dataset,batchsize,nx,partial,imax,1:nx,shuffle,atype,atype)
end

@propagate_inbounds function Base.iterate(d::ImageDatasetBatch, i=0)     # returns data in d.indices[i+1:i+batchsize]
    if i >= d.imax
        return nothing
    end
    if d.shuffle && i == 0
        d.indices = randperm(d.length)
    end
    nexti = min(i + d.batchsize, d.length)
    ids = d.indices[i+1:nexti]
    pairs = [load(d.image_dataset, idx) for idx in ids]
    xs, ys = first.(pairs), last.(pairs)
    xsize, ysize = size(xs[1]), size(ys[1])
    xs = [reshape(x, 1, xsize...) for x in xs]
    ys = [reshape(y, 1, ysize...) for y in ys]
    xs = reshape(vcat(xs...), length(ids), xsize...)
    ys = reshape(vcat(ys...), length(ids), ysize...)
    xbatch = convert(d.xtype, permutedims(xs, [3, 4, 2, 1]))
    ybatch = convert(d.ytype, permutedims(ys, [2, 1]))
    return ((xbatch,ybatch),nexti)
end

function Base.length(d::ImageDatasetBatch)
    n = d.length / d.batchsize
    d.partial ? ceil(Int,n) : floor(Int,n)
end

function load_test_image(filename, transformation, atype)
    image = transformation(FileIO.load(filename) .|> RGB .|> float)
    xs = reshape(image, 1, size(image)...)
    xs = convert(atype, permutedims(xs, [3, 4, 2, 1]))
    return xs
end

function save_test_image(filename, xs)
    xs = permutedims(xs, [4, 3, 1, 2])
    image = reshape(xs, size(xs)[2:end]...)
    image = convert(Array, image)    
    image = image .* 0.5 .+ 0.5
    image = image .|> N0f8 |> colorview(RGB)
    FileIO.save(filename, image)
end

end