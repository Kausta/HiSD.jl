export load_config, create_dir!, prepare_sub_folder, get_train_datasets

using YAML: load_file
using Base.Iterators: Cycle

function load_config(path)
    load_file(path)
end

function create_dir!(dir)
   if !isdir(dir)
        println("Creating directory: $dir")
        mkpath(dir)
    end  
end

function prepare_sub_folder(out_dir)
    output_directory = joinpath(out_dir, "outputs")
    create_dir!(output_directory)
    image_directory = joinpath(output_directory, "images")
    create_dir!(image_directory)
    checkpoint_directory = joinpath(output_directory, "checkpoints")
    create_dir!(checkpoint_directory)
    logs_directory = joinpath(out_dir, "logs")
    create_dir!(logs_directory)
    return output_directory, image_directory, checkpoint_directory, logs_directory
end

function get_train_dataset(attr, transformation, batchsize, data_root; shuffle=true, atype=Knet.atype())
    dataset = ImageDataset(data_root, attr, transformation)
    return Cycle(minibatch(dataset, batchsize, atype, shuffle=shuffle))
end

function get_train_datasets(config, data_root; atype=Knet.atype())
    tags = config["tags"]
    transformation = Compose(
        ColorJitter(0.1, 0.1, 0.1, 0.1),
        RandomHorizontalFlip(0.5),
        to_tensor,
        Normalize(Float32.([0.5, 0.5, 0.5]), Float32.([0.5, 0.5, 0.5]))
    )
    datasets = [[get_train_dataset(attrib["filename"], transformation, config["batch_size"], data_root, shuffle=true, atype=atype) 
            for attrib in tag["attributes"]] for tag in tags] 
    return datasets
end