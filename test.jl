using Images, Interpolations
using Images.FileIO
using Statistics
using Dates
using TensorBoardLogger, Logging, Random
using FileIO
using CUDA
using Knet

DATA_ROOT = "/kuacc/users/ckorkmaz16/HiSD_out"
CONFIG_FILE = "/kuacc/users/ckorkmaz16/HiSD.jl/configs/celeba-hq.yaml"

include("./core/HiSDCore.jl")
using .HiSDCore

println(CUDA.functional())

DATA_ROOT = "/kuacc/users/ckorkmaz16/HiSD_out"
CONFIG_FILE = "/kuacc/users/ckorkmaz16/HiSD.jl/configs/celeba-hq.yaml"

config = load_config(CONFIG_FILE)
println("Loaded config")

model_dt = "2021-12-10_20-18-35"
MODEL_DIR = joinpath(DATA_ROOT, "models", model_dt)
CHECKPOINT_FILE = joinpath(MODEL_DIR, "checkpoint.jld2")

trainer, PREV_ITERS = Knet.load(CHECKPOINT_FILE, "trainer", "iters")

function inference(model, x, steps, transform, noise_dim)
    c = encode(model, x)
    c_trg = c
    for step in steps
        if step["type"] == "latent-guided"
            z = convert(KnetArray{Float32}, randn(Float32, noise_dim, 1))
            s_trg = HiSDCore.map(model, z, step["tag"], step["attribute"])
        elseif step["type"] == "reference-guided"
            reference_idx = rand(1:length(step["reference"]))
            img = load_test_image(step["reference"][reference_idx], transform, Knet.atype())
            s_trg = extract(model, img, step["tag"])
        end
        c_trg = translate(model, c_trg, s_trg, step["tag"])
    end
    x_trg = decode(model, c_trg)
    x_trg
end

function test(model, config, steps, input_path, output_path)
    transform = Compose(
        to_tensor,
        Normalize(Float32.([0.5, 0.5, 0.5]), Float32.([0.5, 0.5, 0.5]))
    )
    if isfile(input_path)
        images = [input_path]
    else
        images = [joinpath(input_path, filename) for filename in readdir(input_path)]
    end
    for filename in images
        image = load_test_image(filename, transform, Knet.atype())
        mapped = inference(model, image, steps, transform, config["noise_dim"])
        save_test_image(joinpath(output_path, "$(basename(filename))_output.jpg"), mapped)
    end
end

curr_dt = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
OUTPUT_DIR = joinpath(DATA_ROOT, "outputs", "model_$(model_dt)", "e$(PREV_ITERS)_$(curr_dt)")
LATENT_REALISM_DIR = joinpath(OUTPUT_DIR, "latent_realism")
LATENT_DISENTANGLEMENT_DIR = joinpath(OUTPUT_DIR, "latent_disentanglement")

DATASET_PATH = joinpath(DATA_ROOT, "datasets_test")
WITHOUT_BANGS = joinpath(DATASET_PATH, "Bangs_without.txt")
WITH_BANGS = joinpath(DATASET_PATH, "Bangs_with.txt")
WITHOUT_BANGS_OUT = joinpath(LATENT_REALISM_DIR, "without_bangs")
WITH_BANGS_OUT = joinpath(LATENT_REALISM_DIR, "with_bangs")

function copy_images(path, to_dir)
    lines = readlines(path)
    lines = (rstrip(line) |> split for line in lines)
    filenames = [line[1] for line in lines]
    create_dir!(to_dir)
    for filename in filenames
        cp(filename, joinpath(to_dir, basename(filename)))
    end
end

copy_images(WITHOUT_BANGS, WITHOUT_BANGS_OUT)
copy_images(WITH_BANGS, WITH_BANGS_OUT)

TEST_IN_PATH = joinpath(LATENT_REALISM_DIR, "without_bangs")
for i in 1:5
    test_out_path =  joinpath(LATENT_REALISM_DIR, "out_$(i)")
    create_dir!(test_out_path)
    steps = [Dict("type" => "latent-guided", "tag" => 1, "attribute" => 1)]
    test(trainer.gen_test, config, steps, TEST_IN_PATH, test_out_path)
end

WITHOUT_BANGS_YM_OUT = joinpath(LATENT_DISENTANGLEMENT_DIR, "without_bangs")
WITH_BANGS_YM_OUT = joinpath(LATENT_DISENTANGLEMENT_DIR, "with_bangs")

function copy_images_young_male(path, to_dir)
    lines = readlines(path)
    lines = (rstrip(line) |> split for line in lines)
    parsed = [(line[1],parse(Int, line[2]),parse(Int, line[3])) for line in lines]
    create_dir!(to_dir)
    for (filename, gender, age) in parsed
        if gender == 1 && age == 1
            cp(filename, joinpath(to_dir, basename(filename)))
        end
    end
end

copy_images_young_male(WITHOUT_BANGS, WITHOUT_BANGS_YM_OUT)
copy_images_young_male(WITH_BANGS, WITH_BANGS_YM_OUT)

TEST_IN_PATH = joinpath(LATENT_DISENTANGLEMENT_DIR, "without_bangs")
for i in 1:5
    test_out_path =  joinpath(LATENT_DISENTANGLEMENT_DIR, "out_$(i)")
    create_dir!(test_out_path)
    steps = [Dict("type" => "latent-guided", "tag" => 1, "attribute" => 1)]
    test(trainer.gen_test, config, steps, TEST_IN_PATH, test_out_path)
end

REFERENCE_REALISM_DIR = joinpath(OUTPUT_DIR, "reference_realism")
REFERENCE_DISENTANGLEMENT_DIR = joinpath(OUTPUT_DIR, "reference_disentanglement")

DATASET_TRAIN_PATH = joinpath(DATA_ROOT, "datasets", "Bangs_with.txt")
DATASET_TEST_PATH = joinpath(DATA_ROOT, "datasets_test", "Bangs_with.txt")
DATASET_PATHS = [DATASET_TRAIN_PATH, DATASET_TEST_PATH]
references_with_bangs = []
for dataset in DATASET_PATHS
    lines = readlines(dataset)
    lines = (rstrip(line) |> split for line in lines)
    filenames = [line[1] for line in lines]
    for filename in filenames
        push!(references_with_bangs, filename)
    end
end

DATASET_PATH = joinpath(DATA_ROOT, "datasets_test")
WITHOUT_BANGS = joinpath(DATASET_PATH, "Bangs_without.txt")
WITH_BANGS = joinpath(DATASET_PATH, "Bangs_with.txt")
WITHOUT_BANGS_OUT = joinpath(REFERENCE_REALISM_DIR, "without_bangs")
WITH_BANGS_OUT = joinpath(REFERENCE_REALISM_DIR, "with_bangs")

function copy_images(path, to_dir)
    lines = readlines(path)
    lines = (rstrip(line) |> split for line in lines)
    filenames = [line[1] for line in lines]
    create_dir!(to_dir)
    for filename in filenames
        cp(filename, joinpath(to_dir, basename(filename)))
    end
end

copy_images(WITHOUT_BANGS, WITHOUT_BANGS_OUT)
copy_images(WITH_BANGS, WITH_BANGS_OUT)

TEST_IN_PATH = joinpath(REFERENCE_REALISM_DIR, "without_bangs")
for i in 1:5
    test_out_path =  joinpath(REFERENCE_REALISM_DIR, "out_$(i)")
    create_dir!(test_out_path)
    steps = [Dict("type" => "reference-guided", "tag" => 1, "reference" => references_with_bangs)]
    test(trainer.gen_test, config, steps, TEST_IN_PATH, test_out_path)
end

WITHOUT_BANGS_YM_OUT = joinpath(REFERENCE_DISENTANGLEMENT_DIR, "without_bangs")
WITH_BANGS_YM_OUT = joinpath(REFERENCE_DISENTANGLEMENT_DIR, "with_bangs")

function copy_images_young_male(path, to_dir)
    lines = readlines(path)
    lines = (rstrip(line) |> split for line in lines)
    parsed = [(line[1],parse(Int, line[2]),parse(Int, line[3])) for line in lines]
    create_dir!(to_dir)
    for (filename, gender, age) in parsed
        if gender == 1 && age == 1
            cp(filename, joinpath(to_dir, basename(filename)))
        end
    end
end

copy_images_young_male(WITHOUT_BANGS, WITHOUT_BANGS_YM_OUT)
copy_images_young_male(WITH_BANGS, WITH_BANGS_YM_OUT)

TEST_IN_PATH = joinpath(REFERENCE_DISENTANGLEMENT_DIR, "without_bangs")
for i in 1:5
    test_out_path =  joinpath(REFERENCE_DISENTANGLEMENT_DIR, "out_$(i)")
    create_dir!(test_out_path)
    steps = [Dict("type" => "reference-guided", "tag" => 1, "reference" => references_with_bangs)]
    test(trainer.gen_test, config, steps, TEST_IN_PATH, test_out_path)
end