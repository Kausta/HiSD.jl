module Data

using Images.FileIO

export ImageData, ImageDataset

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

end