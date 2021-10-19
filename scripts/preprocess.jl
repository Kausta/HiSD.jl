#=
Preprocessing script for categorizing CelebAMask-HQ images by bangs, eyeglasses and hair

Implementation is based on the original preprocessing in HiSD
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
            help = "Start of the training set"
            arg_type = Int
            default = 3002
        "--end"
            help = "End of the training set"
            arg_type = Int
            default = 30002
    end

    return parse_args(settings)
end

# Indexes for different attributes inside CelebAMask-HQ label annonations file
const BANGS_IDX = 7
const BLACKHAIR_IDX = 10
const BLONDEHAIR_IDX = 11
const BROWNHAIR_IDX = 13
const GRAYHAIR_IDX = 19
const EYEGLASSES_IDX = 17
const GENDER_IDX = 22
const AGE_IDX = 41

"""
    write_entry(target_path::String, filename::String, entry::String)

Appends the `entry` with a newline to the file named `filename` at the `target_path`
"""
function write_entry(target_path::String, filename::String, entry::String)
    open(joinpath(target_path, filename), "a") do f
        println(f, entry)
    end
end

"""
    main()

Parses the command line arguments and preprocesses CelebAMask-HQ data
"""
function main()
    parsed_args = parse_commandline()

    target_path = parsed_args["target_path"]
    mkpath(target_path)

    tags_attributes = Dict(
        "Bangs" => ["with", "without"],
        "Eyeglasses" => ["with", "without"],
        "HairColor" => ["black", "blond", "brown"]
    )

    for (tag, attributes) in tags_attributes
        for attribute in attributes
            open(joinpath(target_path, "$(tag)_$(attribute).txt"), "w") do f
            end 
        end
    end

    celeba_imgs = parsed_args["img_path"]
    celeba_label = parsed_args["label_path"]
    start_ind, end_ind = parsed_args["start"], parsed_args["end"]

    lines = readlines(celeba_label)
    lines = lines[start_ind+1:end_ind]

    @showprogress for line in lines 
        line = split(line)
        filename = joinpath(abspath(celeba_imgs), line[1])

        # Use only gender and age as tag-irrelevant conditions. Add other labels if you want.
        entry = "$(filename) $(line[GENDER_IDX]) $(line[AGE_IDX])"

        bangs = parse(Int, line[BANGS_IDX])
        if bangs == 1
            write_entry(target_path, "Bangs_with.txt", entry)
        elseif bangs == -1
            write_entry(target_path, "Bangs_without.txt", entry)
        end

        eyeglasses = parse(Int, line[EYEGLASSES_IDX])
        if eyeglasses == 1
            write_entry(target_path, "Eyeglasses_with.txt", entry)
        elseif eyeglasses == -1
            write_entry(target_path, "Eyeglasses_without.txt", entry)
        end
        
        blackhair = parse(Int, line[BLACKHAIR_IDX])
        blondehair = parse(Int, line[BLONDEHAIR_IDX])
        brownhair = parse(Int, line[BROWNHAIR_IDX])
        grayhair = parse(Int, line[GRAYHAIR_IDX])
        if blackhair == 1 && blondehair == -1 && brownhair == -1 && grayhair == -1
            write_entry(target_path, "HairColor_black.txt", entry)
        elseif blackhair == -1 && blondehair == 1 && brownhair == -1 && grayhair == -1
            write_entry(target_path, "HairColor_blond.txt", entry)
        elseif blackhair == -1 && blondehair == -1 && brownhair == 1 && grayhair == -1
            write_entry(target_path, "HairColor_brown.txt", entry)
        end
    end
end

main() 