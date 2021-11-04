#=
Resizing script CelebAMask-HQ images, to resize everything to 128x128 from 1024x1024 beforehand to load data faster
Resaves as png to lose information from jpg again

Resizing uses the same code from torchvision=0.2.2, same as original HiSD.jl, while using the current PIL version
=#
using ArgParse
using ProgressMeter
using YAML: load_file
using PyCall

@pyimport PIL.Image as Image

"""
    parse_commandline()

Processes the command line arguments for CelebAMask-HQ preprocessing
"""
function parse_commandline()
    settings = ArgParseSettings()

    @add_arg_table! settings begin
        "--img_path"
            help = "Path to CelebA-HQ Images"
            arg_type = String
            required = true
        "--config"
            help = "Path to training config for sizes"
            arg_type = String
            required = true
    end

    return parse_args(settings)
end

function resize(image, size; interpolation=Image.BILINEAR)
    if isa(size, Number)
        w, h = image.size
        if (w <= h &&  w == size) || (h <= w && h == size)
            return image
        end

        oh, ow = if w < h
            Int(size * h / w), size
        else
            size, Int(size * w / h) 
        end
    else
        oh, ow = size
    end

    return image.resize((ow, oh), interpolation)
end

"""
    main()

Parses the command line arguments and preprocesses CelebAMask-HQ data
"""
function main()
    parsed_args = parse_commandline()
    
    config = load_file(parsed_args["config"])
    size = config["new_size"]
    println("Resizing to new size $(size)x$(size)")
    
    img_path = parsed_args["img_path"]
    # Read jpg files from the dir
    files = readdir(img_path)
    files = filter(x -> endswith(x, ".jpg"), files)

    @showprogress for file in files
        image = Image.open(joinpath(img_path, file))
        
        resized = resize(image, size)
        
        new_path = joinpath(img_path, replace(file, ".jpg" => ".png"))
        resized.save(new_path, "PNG")
    end
end

main() 