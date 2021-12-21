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

USE_CHECKPOINT=true
# curr_dt = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
curr_dt = "2021-12-10_20-18-35"

MODEL_DIR = joinpath(DATA_ROOT, "models", curr_dt)
create_dir!(MODEL_DIR)
IMAGE_DIR = joinpath(MODEL_DIR, "images")
create_dir!(IMAGE_DIR)
CHECKPOINT_FILE = joinpath(MODEL_DIR, "checkpoint.jld2")

config = load_config(CONFIG_FILE)
println("Loaded config")

datasets = get_train_datasets(config, DATA_ROOT, atype=KnetArray{Float32})
println("Loaded $(sum(length.(datasets))) datasets")
flush(stdout) 

function get_trainer()
    if USE_CHECKPOINT
        trainer, PREV_ITERS = Knet.load(CHECKPOINT_FILE, "trainer", "iters")
        println("Loaded model")
        return trainer, PREV_ITERS + 1
    else
        trainer = HiSDTrainer(config)
        println("Initialized model")
        Knet.save(CHECKPOINT_FILE, "trainer", trainer, "iters", 1)
        return trainer, 1
    end
end

trainer, prev_iters = get_trainer()
flush(stdout) 

tags = collect(1:length(datasets))
data_states = [zeros(Int32, length(x)) for x in datasets]

if prev_iters == 1
    for i in 1:length(datasets)
        j, j_trg = randperm(length(datasets[i]))[1:2]
        
        (x,_),data_state = iterate(datasets[i][j], data_states[i][j])
        data_states[i][j] = data_state

        (x_trg,_),data_state = iterate(datasets[i][j_trg], data_states[i][j_trg])
        data_states[i][j_trg] = data_state

        test_image_outputs = sample(trainer, x, x_trg, j, j_trg, i)
        write_2images(test_image_outputs,
                      config["batch_size"], 
                      IMAGE_DIR, 
                      "sample_$(1)_$(config["tags"][i]["name"])_$(config["tags"][i]["attributes"][j]["name"])_to_$(config["tags"][i]["attributes"][j_trg]["name"])")
    end
end

start_time = Dates.now()
for iter in prev_iters:config["total_iterations"]
    i = rand(tags)
    j, j_trg = randperm(length(datasets[i]))[1:2]
    (x,y),data_state = iterate(datasets[i][j], data_states[i][j])
    data_states[i][j] = data_state
    G_adv, G_sty, G_rec, D_adv = update(trainer, x, y, i, j, j_trg)
    now = Dates.now()
    
    if iter % config["image_save_iter"] == 0
        for i in 1:length(datasets)
            j, j_trg = randperm(length(datasets[i]))[1:2]
            
            (x,_),data_state = iterate(datasets[i][j], data_states[i][j])
            data_states[i][j] = data_state

            (x_trg,_),data_state = iterate(datasets[i][j_trg], data_states[i][j_trg])
            data_states[i][j_trg] = data_state

            test_image_outputs = sample(trainer, x, x_trg, j, j_trg, i)
            write_2images(test_image_outputs,
                          config["batch_size"], 
                          IMAGE_DIR, 
                          "sample_$(iter+1)_$(config["tags"][i]["name"])_$(config["tags"][i]["attributes"][j]["name"])_to_$(config["tags"][i]["attributes"][j_trg]["name"])")
        end
    end

    if iter % config["log_iter"] == 0
        println("Iter $iter")
        println("($(now-start_time)): $G_adv $G_sty $G_rec $D_adv")
        flush(stdout) 
        global start_time = now
    end

    if iter % config["snapshot_save_iter"] == 0
        Knet.save(CHECKPOINT_FILE, "trainer", trainer, "iters", iter)
    end
end