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
CONFIG_FILE = "/kuacc/users/ckorkmaz16/HiSD.jl/configs/celeba-hq-test.yaml"

config = load_config(CONFIG_FILE)
println("Loaded config")

datasets = get_train_datasets(config, DATA_ROOT, atype=KnetArray{Float32})
println("Loaded $(sum(length.(datasets))) datasets")

function generator_iter(tr, x, y)
    outs = GenOutputs()
    CUDA.@time loss_gen_total = @diff tr(outs, x, y, 1, 1, 2, mode="gen")
    outs = to_val(outs)
    @show loss_gen_total
    loss_gen_total = nothing
    GC.gc(true)
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
    outs = generator_iter(tr, x, y)
    discriminator_iter(tr, x, y, outs)
end

trainer = HiSDTrainer(config)
Knet.save((joinpath(DATA_ROOT, "outputs", "checkpoint.jld2")), "trainer", trainer)

tags = collect(1:length(datasets))
data_states = [zeros(Int32, length(x)) for x in datasets]
start_time = Dates.now()
for iter in 1:config["total_iterations"]
    i = rand(tags)
    j = rand(collect(1:length(datasets[i])))
    j_trg = rand(collect(1:length(datasets[i])))
    (x,y),data_state = iterate(datasets[i][j], data_states[i][j])
    data_states[i][j] = data_state
    G_adv, G_sty, G_rec, D_adv = update(trainer, x, y, i, j, j_trg)
    now = Dates.now()
    
    if iter % config["log_iter"] == 0
        println("Iter $iter")
        println("($(now-start_time)): $G_adv $G_sty $G_rec $D_adv")
        global start_time = now
    end

    if iter % config["snapshot_save_iter"] == 0
        Knet.save((joinpath(DATA_ROOT, "outputs", "checkpoint.jld2")), "trainer", trainer)
    end
end