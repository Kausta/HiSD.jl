#=
=#
using ArgParse
using ProgressMeter

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
        "--label_path"
            help = "Path to CelebA-HQ Labels"
            arg_type = String
            required = true
        "--target_path"
            help = "Output directory"
            arg_type = String
            required = true
        "--start"
            help = "Start of the test set"
            arg_type = Int
            default = 2
        "--end"
            help = "End of the test set"
            arg_type = Int
            default = 3002
        "--extension"
            help = "Extension of images to copy"
            arg_type = String
            default = ".jpg"
    end

    return parse_args(settings)
end


"""
    main()

Parses the command line arguments and preprocesses CelebAMask-HQ data
"""
function main()
    parsed_args = parse_commandline()

    target_path = parsed_args["target_path"]
    mkpath(target_path)

    celeba_imgs = parsed_args["img_path"]
    celeba_label = parsed_args["label_path"]
    start_ind, end_ind = parsed_args["start"], parsed_args["end"]
    extension = parsed_args["extension"]

    lines = readlines(celeba_label)
    lines = lines[start_ind+1:end_ind]

    @showprogress for line in lines 
        line = split(line)
        filename_base = line[1]
        if extension != ".jpg"
            filename_base = replace(filename_base, ".jpg" => extension)
        end
        filename = joinpath(abspath(celeba_imgs), filename_base)
        out_filename = joinpath(abspath(target_path), filename_base)
        cp(filename, out_filename)
    end
end

main() 