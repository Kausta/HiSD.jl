module HiSDCore

using CUDA: CUDA, CuArray
using Knet

atype=KnetArray{Float32}
# atype=Knet.atype()

include("data.jl")
include("transformations.jl")
include("utils.jl")
include("primitives.jl")
include("networks.jl")
include("trainer.jl")

end