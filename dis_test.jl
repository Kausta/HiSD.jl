using Images, Interpolations
using Images.FileIO
using Statistics
using Dates
using TensorBoardLogger, Logging, Random
using FileIO
using CUDA: CUDA, CuArray
using Knet

include("./core/HiSDCore.jl")
using .HiSDCore

println(CUDA.functional())

DATA_ROOT = "/kuacc/users/ckorkmaz16/HiSD_out"
CONFIG_FILE = "/kuacc/users/ckorkmaz16/HiSD.jl/configs/celeba-hq.yaml"

config = load_config(CONFIG_FILE)
println("Loaded config")

datasets = get_train_datasets(config, DATA_ROOT, atype=KnetArray{Float32})
println("Loaded $(sum(length.(datasets))) datasets")

function generator_out(tr, x, y)
    outs = GenOutputs()
    loss_gen_total = tr(outs, x, y, 1, 1, 2, mode="gen")
    return outs
end

function discriminator_iter(tr, x, y, outs)
    CUDA.@time loss_dis_total = @diff tr(x, outs.x_trg, outs.x_cyc, y, 1, 1, 2, mode="dis")
    @show loss_dis_total
end

function test_single_iter()
    (x,y),i = iterate(datasets[1][1], 0)
    @show size(x), size(y), i, typeof(x), typeof(y)

    tr = HiSD(config)
    println("Initialized model")
    outs = generator_out(tr, x, y)
    discriminator_iter(tr, x, y, GenOutputs(0.0, 0.0, 0.0, x, x))
end

test_single_iter()