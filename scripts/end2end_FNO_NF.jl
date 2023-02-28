## A 2D compass example

using DrWatson
@quickactivate "jutul-compass"

using Pkg; Pkg.instantiate();
include(srcdir("dummy_src_file.jl"))
using JUDI
dummy_JUDI_operation()
using JutulDarcyAD
using LinearAlgebra
using PyPlot
using Flux
using LineSearches
using JLD2
using Statistics
using Images
using Random
using InvertibleNetworks
using GlowInvertibleNetwork
using FNO4CO2

Random.seed!(2023)

if gpu_flag
    device = gpu
else
    device = cpu
end

matplotlib.use("agg")

sim_name = "end2end-inv"
exp_name = "compass-FNO-NF"

mkpath(datadir())
mkpath(plotsdir())

## load FNO
net_path_FNO = datadir("3D_FNO", "batch_size=8_dt=0.058824_ep=400_epochs=1000_learning_rate=0.0001_nt=18_ntrain=2400_nvalid=40_s=1_width=20.jld2")
mkpath(datadir("3D_FNO"))

# Download the dataset into the data directory if it does not exist
if ~isfile(net_path_FNO )
    run(`wget https://www.dropbox.com/s/3r3ol1gnblh3slf/'
        'batch_size=8_dt=0.058824_ep=400_epochs=1000_learning_rate=0.0001_nt=18_ntrain=2400_nvalid=40_s=1_width=20.jld2 -q -O $net_path_FNO`)
end

net_dict_FNO = JLD2.jldopen(net_path_FNO, "r")
NN = net_dict_FNO["NN_save"];
AN = net_dict_FNO["AN"];
grid = gen_grid(net_dict_FNO["n"], net_dict_FNO["d"], net_dict_FNO["nt"], net_dict_FNO["dt"]);
Flux.testmode!(NN, true);

function perm_to_tensor_(x_perm::AbstractMatrix{Float32},grid::Array{Float32,4},AN::ActNorm)
    # input nx*ny, output nx*ny*nt*4*1
    nx, ny, nt, _ = size(grid)
    return cat(reshape(cat([AN.s.data[1].*(reshape(x_perm, nx, ny, 1, 1))[:,:,1,1].+AN.b.data for i = 1:nt]..., dims=3), nx, ny, nt, 1, 1),
    reshape(grid, nx, ny, nt, 3, 1), dims=4)
end

function S(x)
    return clamp.(NN(perm_to_tensor_(x, grid, AN)), 0f0, 0.9f0)[:,:,1:15,1];
end

## load NF
net_path = datadir("trained-NF", "clip_norm=5.0_depth=5_e=30_lr=0.0002_nc_hidden=512_nscales=4_ntrain=10000_nx=128_ny=80_α=0.1_αmin=0.01.jld2")
mkpath(datadir("trained-NF"))

# Download the dataset into the data directory if it does not exist
if ~isfile(net_path)
    run(`wget https://www.dropbox.com/s/fc22lk28u5z2d04/'
        'clip_norm=5.0_depth=5_e=30_lr=0.0002_nc_hidden=512_nscales=4_ntrain=10000_nx=128_ny=80_α=0.1_αmin=0.01.jld2 -q -O $net_path`)
end

network_dict = JLD2.jldopen(net_path, "r");
Params = network_dict["Params"];
Rrn_here = network_dict["Rrn_here"];

copy!(Random.default_rng(), Rrn_here);
opts = GlowOptions(; cl_activation=SigmoidNewLayer(0.5f0),
                    cl_affine=true,
                    init_cl_id=true,
                    conv1x1_nvp=false,
                    init_conv1x1_permutation=true,
                    conv1x1_orth_fixed=true,
                    T=Float32)
G = Glow(1, network_dict["nc_hidden"], network_dict["depth"], network_dict["nscales"]; logdet=false, opt=opts) #|> gpu
set_params!(G, Params);

# check generative samples are good so that loading went well. 
G.forward(randn(Float32,network_dict["nx"],network_dict["ny"],1,1));
gen = G.inverse(randn(Float32,network_dict["nx"],network_dict["ny"],1,1));
# generator now
G = reverse(G);

## grid size
JLD2.@load datadir("BGCompass_tti_625m.jld2") m d rho;
vp = 1f0./sqrt.(m)
d = (6., 6.)
n = size(m)

# downsample
cut_area = [201, 456, 182, n[end]]
m = m[cut_area[1]:cut_area[2],cut_area[3]:cut_area[4]]
h = (cut_area[3]-1) * d[end]
v = Float64.(sqrt.(1f0./m));
factor = 2
v = 1.0./downsample(1.0./v, factor)

## flow dimension
ns = (size(v,1), 1, size(v,2))
ds = (d[1] * factor, 2000.0, d[2] * factor)
Kh = VtoK(v, (ds[1], ds[end]));
K = Float64.(Kh * md);

# set up jutul model
ϕ = 0.25
model = jutulModel(ns, ds, ϕ, K1to3(K; kvoverkh=0.36); h=h)

## simulation time steppings
tstep = 365.25 * ones(15)
tot_time = sum(tstep)

## injection & production
inj_loc = (Int(round(ns[1]/2)), 1, ns[end]-20) .* ds
irate = 0.3
f = jutulForce(irate, [inj_loc])

## set up modeling operator
Sjutul = jutulModeling(model, tstep)

## simulation
mesh = CartesianMesh(model)
T(x) = log.(KtoTrans(mesh, K1to3(exp.(x); kvoverkh=0.36)))

logK = log.(K)

mkpath(datadir("flow-data"))
if ~isfile(datadir("flow-data", "true_state.jld2"))
    run(`wget https://www.dropbox.com/s/ts2wntxnqy1zqdr/'
        'true_state.jld2 -q -O $(datadir("flow-data", "true_state.jld2"))`)
end
JLD2.@load datadir("flow-data", "true_state.jld2") state

### observed states
nv = length(tstep)
O(state::AbstractArray) = [state[:,:,i] for i = 1:nv]
function O(state::jutulStates)
    full_his = Float32.(reshape(state[1:nv*prod(ns)], ns[1], ns[end], nv))
    return [full_his[:,:,i] for i = 1:nv]
end
sw_true = O(state)

function CO2mute(sw_true::Vector{Matrix{Float32}}; clip::Float32=2f-3)
    sw_smooth = [imfilter(sw_true[i], Kernel.gaussian(5)) for i = 1:length(sw_true)];
    mask = [Float32.(imfilter(Float32.(sw_smooth[i] .>= clip), Kernel.gaussian(5))) for i = 1:length(sw_smooth)]
    return mask
end
mask = CO2mute(sw_true);
known_idx = [findlast(Kh[i,:] .== 1e-3) for i = 1:size(Kh,1)]
known_mask = ones(Float32, size(Kh))
for i = 1:size(known_mask,1)
    known_mask[i,1:known_idx[i]] .= 0f0
end
mask = [mask[i] .* known_mask for i = 1:length(mask)]
M(sw::Vector{Matrix{Float32}}) = [sw[i] .* mask[i] for i = 1:length(sw)]

### pad co2 back to normal size
pad(c::Matrix{Float32}) =
    hcat(zeros(Float32, n[1], cut_area[3]-1),
    vcat(zeros(Float32, cut_area[1]-1, factor * size(c,2)), repeat(c, inner=(factor, factor)), zeros(Float32, n[1]-cut_area[2], factor * size(c,2))))
pad(c::Vector{Matrix{Float32}}) = [pad(c[i]) for i = 1:nv]

sw_pad = pad(sw_true)

# set up rock physics
phi = Float32(ϕ) * ones(Float32,n)  # porosity
R(c::Vector{Matrix{Float32}}) = Patchy(c,1f3*vp,1f3*rho,phi; bulk_min = 5f10)[1]/1f3
vps = R(sw_pad)   # time-varying vp

##### Wave equation
o = (0f0, 0f0)          # origin

nsrc = 32       # num of sources
nrec = 960      # num of receivers

models = [Model(n, d, o, (1f0 ./ vps[i]).^2f0; nb = 80) for i = 1:nv]   # wave model

timeS = timeR = 3600f0               # recording time
dtS = dtR = 4f0                     # recording time sampling rate
ntS = Int(floor(timeS/dtS))+1       # time samples
ntR = Int(floor(timeR/dtR))+1       # source time samples
idx_wb = minimum(find_water_bottom(vp.-minimum(vp)))

mode = "transmission"
if mode == "reflection"
    xsrc = convertToCell(range(d[1],stop=(n[1]-1)*d[1],length=nsrc))
    zsrc = convertToCell(range(10f0,stop=10f0,length=nsrc))
    xrec = range(d[1],stop=(n[1]-1)*d[1],length=nrec)
    zrec = range((idx_wb-1)*d[2]-2f0,stop=(idx_wb-1)*d[2]-2f0,length=nrec)
elseif mode == "transmission"
    xsrc = convertToCell(range(d[1],stop=d[1],length=nsrc))
    zsrc = convertToCell(range((idx_wb-1)*d[2]+10f0,stop=(n[2]-1)*d[2],length=nsrc))
    xrec = range((n[1]-1)*d[1],stop=(n[1]-1)*d[1], length=nrec)
    zrec = range((idx_wb-1)*d[2]+10f0,stop=(n[2]-1)*d[2],length=nrec)
else
    # source locations -- half at the left hand side of the model, half on top
    xsrc = convertToCell(vcat(range(d[1],stop=d[1],length=Int(nsrc/2)),range(d[1],stop=(n[1]-1)*d[1],length=Int(nsrc/2))))
    zsrc = convertToCell(vcat(range((idx_wb-1)*d[2]+10f0,stop=(n[2]-1)*d[2],length=Int(nsrc/2)),range(10f0,stop=10f0,length=Int(nsrc/2))))
    xrec = vcat(range((n[1]-1)*d[1],stop=(n[1]-1)*d[1], length=Int(nrec/2)),range(d[1],stop=(n[1]-1)*d[1],length=Int(nrec/2)))
    zrec = vcat(range((idx_wb-1)*d[2]+10f0,stop=(n[2]-1)*d[2],length=Int(nrec/2)),range(10f0,stop=10f0,length=Int(nrec/2)))
end

ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
yrec = 0f0

# set up src/rec geometry
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# set up source
f0 = 0.02f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

# set up simulation operators
Fs = [judiModeling(models[i], srcGeometry, recGeometry) for i = 1:nv] # acoustic wave equation solver

## wave physics
function F(v::Vector{Matrix{Float32}})
    m = [vec(1f0./v[i]).^2f0 for i = 1:nv]
    return [Fs[i](m[i], q) for i = 1:nv]
end

# Define seismic data directory
mkpath(datadir("seismic-data"))
misc_dict = @strdict nsrc nrec nv f0 cut_area tstep factor d n

### generate/load data
if ~isfile(datadir("seismic-data", savename(misc_dict, "jld2"; digits=6)))
    println("generating data")
    global d_obs = [Fs[i]*q for i = 1:nv]
    seismic_dict = @strdict nsrc nrec nv f0 cut_area tstep factor d n d_obs q srcGeometry recGeometry model
    @tagsave(
        datadir("seismic-data", savename(seismic_dict, "jld2"; digits=6)),
        seismic_dict;
        safe=true
    )
else
    println("loading data")
    JLD2.@load datadir("seismic-data", savename(misc_dict, "jld2"; digits=6)) d_obs
    global d_obs = d_obs
end

# Main loop
niterations = 200
fhistory = zeros(niterations)

### inversion initialization
logK0 = deepcopy(logK)
logK0[v.>3.5] .= mean(logK[v.>3.5])
logK0 = Float32.(logK0)
z = G.inverse(reshape(logK0, ns[1], ns[end], 1, 1))
logK_init = deepcopy(logK0)
y_init = box_co2(M(O(S(logK_init))))

# GD algorithm
learning_rate = 1f0
lr_min = learning_rate*1f-2
nssample = 4
nbatches = div(nsrc, nssample)
decay_rate = exp(log(lr_min/learning_rate)/niterations)
opt = Flux.Optimiser(ExpDecay(learning_rate, decay_rate, 1, lr_min), Descent(1f0))
λ = 0f0

for j=1:niterations

    Base.flush(Base.stdout)

    ### subsample sources
    rand_ns = [jitter(nsrc, nssample) for i = 1:nv]                             # select random source idx for each vintage
    q_sub = [q[rand_ns[i]] for i = 1:nv]                                        # set-up source
    F_sub = [Fs[i][rand_ns[i]] for i = 1:nv]                                 # set-up wave modeling operator
    dobs = [d_obs[i][rand_ns[i]] for i = 1:nv]                                  # subsampled seismic dataset from the selected sources
    function F(v::Vector{Matrix{Float32}})
        m = [vec(1f0./v[i]).^2f0 for i = 1:nv]
        return [F_sub[i](m[i], q_sub[i]) for i = 1:nv]
    end

    # objective function for inversion
    function obj(z)
        global logK_j = box_logK(G(z)[:,:,1,1])
        global c_j = box_co2(M(O(S(logK_j))))
        global dpred_j = F(box_v(R(pad(c_j))))
        fval = .5f0 * norm(dpred_j-dobs)^2f0/nssample/nv + .5f0 * λ^2f0 * norm(z)^2f0/length(z) 
        @show fval
        return fval
    end
    ## AD by Flux
    @time fval, gs = Flux.withgradient(() -> obj(z), Flux.params(z))
    g = gs[z]
    Flux.Optimise.update!(opt, z, g)
    fhistory[j] = fval
    println("Inversion iteration no: ",j,"; function value: ", fhistory[j])

    ### save intermediate results
    save_dict = @strdict j λ z nssample f0 logK_j g step niterations nv nsrc nrec nv cut_area tstep factor n d fhistory mask
    @tagsave(
        joinpath(datadir(sim_name, exp_name), savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

    ## save figure
    fig_name = @strdict j λ nssample f0 dlogK logK_j g niterations nv nsrc nrec nv cut_area tstep factor n d fhistory mask

    ## compute true and plot
    SNR = -2f1 * log10(norm(K-exp.(logK_j))/norm(K))
    fig = figure(figsize=(20,12));
    subplot(2,2,1);
    imshow(exp.(logK_j)'./md,vmin=0,vmax=maximum(K/md));title("inversion by NN, $(j) iter");colorbar();
    subplot(2,2,2);
    imshow(K'./md,vmin=0,vmax=maximum(K/md));title("GT permeability");colorbar();
    subplot(2,2,3);
    imshow(exp.(logK_init)'./md,vmin=0,vmax=maximum(K/md));title("initial permeability");colorbar();
    subplot(2,2,4);
    imshow(abs.(K'-exp.(logK_j)')./md,vmin=0,vmax=maximum(K/md));title("abs error, SNR=$SNR");colorbar();
    suptitle("End-to-end Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(save_dict; digits=6)*"_K.png"), fig);
    close(fig)

    ## loss
    fig = figure(figsize=(20,12));
    plot(fhistory[1:j]);title("loss=$(fhistory[j])");
    suptitle("End-to-end Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(save_dict; digits=6)*"_loss.png"), fig);
    close(fig)

    ## data fitting
    fig = figure(figsize=(20,12));
    for i = 1:5
        subplot(4,5,i);
        imshow(y_init[3*i]', vmin=0, vmax=1);
        title("initial prediction at snapshot $(3*i)")
        subplot(4,5,i+5);
        imshow(sw_true[3*i]', vmin=0, vmax=1);
        title("true at snapshot $(3*i)")
        subplot(4,5,i+10);
        imshow(c_j[3*i]', vmin=0, vmax=1);
        title("predict at snapshot $(3*i)")
        subplot(4,5,i+15);
        imshow(5*abs.(sw_true[3*i]'-c_j[3*i]'), vmin=0, vmax=1);
        title("5X diff at snapshot $(3*i)")
    end
    suptitle("End-to-end Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(save_dict; digits=6)*"_saturation.png"), fig);
    close(fig)

end
