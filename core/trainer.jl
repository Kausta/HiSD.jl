using Statistics: mean
using LinearAlgebra: norm, lmul!, axpy!

# Must store additional outputs in a mutable struct since Autograd requires a function returning a single scalar value
export GenOutputs, to_val
mutable struct GenOutputs; loss_gen_adv; loss_gen_sty; loss_gen_rec; x_trg; x_cyc; end
GenOutputs() = GenOutputs(nothing, nothing, nothing, nothing, nothing)
to_val(g::GenOutputs) = GenOutputs(value(g.loss_gen_adv), value(g.loss_gen_sty), value(g.loss_gen_rec), value(g.x_trg), value(g.x_cyc))
export HiSD
struct HiSD; gen; dis; noise_dim; config; end
function HiSD(config)
    gen = Generator(config)
    dis = Discriminator(config)
    noise_dim = config["noise_dim"]
    HiSD(gen, dis, noise_dim, config)
end
function (h::HiSD)(args...; mode)
    if mode == "gen"
        return gen_losses(h, args...)
    elseif mode == "dis"
        return dis_losses(h, args...)
    end
end
function gen_losses(h::HiSD, outs, x, y, i, j, j_trg)
    B = size(x)[end]
    
    # non-translation path
    e = encode(h.gen, x)
    x_rec = decode(h.gen, e)
    # println("Non-translation path")

    # self-translation path
    s = extract(h.gen, x, i)
    e_slf = translate(h.gen, e, s, i)
    x_slf = decode(h.gen, e_slf)
    # println("Self-translation path")
    
    # cycle-translation path
    ## translate
    s_trg = map(h.gen, convert(atype, randn(Float32, h.noise_dim, B)), i, j_trg)
    e_trg = translate(h.gen, e, s_trg, i)
    x_trg = decode(h.gen, e_trg)
    ## cycle-back
    e_trg_rec = encode(h.gen, x_trg)
    s_trg_rec = extract(h.gen, x_trg, i) 
    e_cyc = translate(h.gen, e_trg_rec, s, i)
    x_cyc = decode(h.gen, e_cyc)
    # println("Cycle-translation path")
    
    loss_gen_adv = gen_loss_real(h.dis, x, y, i, j) + gen_loss_fake_trg(h.dis, x_trg, y, i, j_trg) + gen_loss_fake_cyc(h.dis, x_cyc, y, i, j)
    
    loss_gen_sty = mean(abs.(s_trg_rec - s_trg))

    loss_gen_rec = mean(abs.(x_rec - x)) + mean(abs.(x_slf - x)) + mean(abs.(x_cyc - x))
    
    loss_gen_total = h.config["adv_w"] * loss_gen_adv + h.config["sty_w"] * loss_gen_sty + h.config["rec_w"] * loss_gen_rec
    # println("Loss calc")

    outs.loss_gen_adv = loss_gen_adv
    outs.loss_gen_sty = loss_gen_sty
    outs.loss_gen_rec = loss_gen_rec
    outs.x_trg = x_trg
    outs.x_cyc = x_cyc
    
    return loss_gen_total
end
function dis_losses(h::HiSD, x, x_trg, x_cyc, y, i, j, j_trg)
    loss = dis_loss_real(h.dis, x, y, i, j) + dis_loss_fake_trg(h.dis, x_trg, y, i, j_trg) + dis_loss_fake_cyc(h.dis, x_cyc, y, i, j)
    return loss
end

clone(a::Adam)=Adam(a.lr,a.beta1,a.beta2,a.eps,0,a.gclip,nothing,nothing)

export HiSDTrainer, update, polyak_average, sample
struct HiSDTrainer; hisd; gen_test; opt_gen_others; opt_gen_mappers; opt_dis; end
function HiSDTrainer(config)
    hisd = HiSD(config)
    beta1 = config["beta1"]
    beta2 = config["beta2"]
    gclip = 100
    opt_gen_others = Adam(lr=config["lr_gen_others"], beta1=beta1, beta2=beta2, gclip=gclip)
    opt_gen_mappers = Adam(lr=config["lr_gen_mappers"], beta1=beta1, beta2=beta2, gclip=gclip)
    opt_dis = Adam(lr=config["lr_dis"], beta1=beta1, beta2=beta2, gclip=gclip)
    gen_test = deepcopy(hisd.gen)

    for p in paramlist_others(hisd.gen)
        p.opt = clone(opt_gen_others)
    end
    for p in paramlist_mappers(hisd.gen)
        p.opt = clone(opt_gen_mappers)
    end
    for p in paramlist(hisd.dis)
        p.opt = clone(opt_dis)
    end

    return HiSDTrainer(hisd, gen_test, opt_gen_others, opt_gen_mappers, opt_dis)
end
function update(trainer::HiSDTrainer, x, y, i, j, j_trg)
    outs = GenOutputs()
    loss_gen_total = @diff trainer.hisd(outs, x, y, i, j, j_trg, mode="gen")
    outs = to_val(outs)
    for p in paramlist(trainer.hisd.gen)
        update!(p, grad(loss_gen_total, p))
    end
    loss_gen_total = nothing
    GC.gc(true)
    loss_dis_total = @diff trainer.hisd(x, outs.x_trg, outs.x_cyc, y, i, j, j_trg, mode="dis")
    for p in paramlist(trainer.hisd.dis)
        update!(p, grad(loss_dis_total, p))
    end
    loss_dis_total = value(loss_dis_total)
    GC.gc(true)
    polyak_average(trainer)
    return outs.loss_gen_adv, outs.loss_gen_sty, outs.loss_gen_rec, loss_dis_total
end
function polyak_average(trainer::HiSDTrainer, beta=0.99)
    gen_params = paramlist(trainer.hisd.gen)
    gen_test_params = paramlist(trainer.gen_test)
    for (gen_p, gen_test_p) in Iterators.zip(gen_params, gen_test_params) 
        # gen_test_p = beta * gen_test_p + (1-beta) * gen_p
        lmul!(beta, gen_test_p.value)
        axpy!(1-beta, gen_p.value, gen_test_p.value)
    end
end
function sample(trainer::HiSDTrainer, x, x_trg, j, j_trg, i)
    B = size(x)[end]

    gen = trainer.gen_test
    out = [x]
    e = encode(gen, x)

    # Latent-guided 1 
    z = convert(atype, repeat(randn(trainer.hisd.noise_dim, 1), 1, B))
    s_trg = map(gen, z, i, j_trg)
    x_trg_ = decode(gen, translate(gen, e, s_trg, i))
    push!(out, x_trg_)

    # Latent-guided 2
    z = convert(atype, repeat(randn(trainer.hisd.noise_dim, 1), 1, B))
    s_trg = map(gen, z, i, j_trg)
    x_trg_ = decode(gen, translate(gen, e, s_trg, i))
    push!(out, x_trg_)

    s_trg = extract(gen, x_trg, i)
    # Reference-guided 1: use x_trg[0, 1, ..., n] as reference
    x_trg_ = decode(gen, translate(gen, e, s_trg, i))
    push!(out, x_trg)
    push!(out, x_trg_)

    # Reference-guided 2: use x_trg[n, n-1, ..., 0] as reference
    x_trg_ = decode(gen, translate(gen, e, s_trg[:,end:-1:1], i))
    push!(out, x_trg[:,:,:,end:-1:1])
    push!(out, x_trg_)

    return out
end