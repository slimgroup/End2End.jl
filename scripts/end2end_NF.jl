## A 2D compass example

using DrWatson
@quickactivate "jutul-compass"

using Pkg; Pkg.instantiate();
nthreads = try
    # Slurm
    parse(Int, ENV["SLURM_CPUS_ON_NODE"])
    using ThreadPinning
    pinthreads(:cores)
catch e
    # Desktop
    Sys.CPU_THREADS
end
include(srcdir("dummy_src_file.jl"))
using JUDI
dummy_JUDI_operation()
using JutulDarcyRules
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
using JSON
Random.seed!(2023)

matplotlib.use("agg")

sim_name = "end2end-inv"
exp_name = "compass-NF"

mkpath(datadir())
mkpath(plotsdir())


## load compass
JLD2.@load datadir("image2023_v_rho.jld2") v rho
vp = deepcopy(v)
n = size(vp)
d = (6f0, 6f0)

cut_area = [1, n[1], 182, n[end]]
v = Float64.(vp[cut_area[1]:cut_area[2], cut_area[3]:cut_area[4]])
factor = (3,2)
d = (6., 6.)
h = 181 * d[end]
v = 1.0./downsample(1.0./v, factor)
ds = Float64.(d) .* factor;

ns = (size(v,1), 1, size(v,2))
ds = (ds[1], ds[1]*ns[1], ds[2])

Kh = VtoK.(v);
K = Float64.(Kh * md);

# set up jutul model
ϕ = 0.25
kvoverkh = 0.36
model = jutulModel(ns, ds, ϕ, K1to3(K; kvoverkh=kvoverkh); h=h)

## simulation time steppings
tstep = 365.25 * 5 * ones(5)
tot_time = sum(tstep)

## injection & production
inj_loc = (128, 1, ns[end]-20) .* ds
pore_volumes = ϕ * sum(v.>3.5) * prod(ds)
irate = 0.2 * pore_volumes / tot_time / 24 / 60 / 60
#irate = 0.3
f = jutulVWell(irate, (inj_loc[1], inj_loc[2]); startz = 46 * ds[end], endz = 48 * ds[end])

## set up modeling operator
S = jutulModeling(model, tstep)

## simulation
mesh = CartesianMesh(model)
T(x) = log.(KtoTrans(mesh, K1to3(exp.(x); kvoverkh=kvoverkh)))

logK = log.(K)

@time state = S(T(logK), f)

### observed states
nv = length(tstep)
function O(state::AbstractVector)
    full_his = Float32.(reshape(state[1:nv*prod(ns)], ns[1], ns[end], nv))
    return [full_his[:,:,i] for i = 1:nv]
end
sw_true = O(state)

known_idx = [findlast(v[i,:].<=3.5) for i = 1:size(Kh,1)]
mask = Vector{Matrix{Float32}}(undef, nv)
for i = 1:nv

    energy_i = [norm(sw_true[i][ix,:]) for ix in axes(sw_true[i],1)]
    first_i = findfirst(energy_i.>0)
    end_i = findlast(energy_i.>0)
    mask_i = zeros(Float32, size(sw_true[i]))
    mask_i[max(1, first_i-5):min(end_i+5, size(sw_true[i],1)), :] .= 1f0
    mask_i = Float32.(imfilter(mask_i, Kernel.gaussian(3)))
    for j in axes(mask_i, 1)
        mask_i[j,1:known_idx[j]] .= 0f0
    end
    mask[i] = mask_i

end
M(sw::Vector{Matrix{Float32}}) = [sw[i] .* mask[i] for i = 1:length(sw)]

### pad co2 back to normal size
pad(c::Matrix{Float32}) =
    hcat(zeros(Float32, n[1], cut_area[3]-1),
    vcat(zeros(Float32, cut_area[1]-1, factor[2] * size(c,2)), repeat(c, inner=factor), zeros(Float32, n[1]-cut_area[2], factor[2] * size(c,2))))
pad(c::Vector{Matrix{Float32}}) = [pad(c[i]) for i = 1:nv]

sw_pad = pad(sw_true)

# set up rock physics
phi = Float32(ϕ) * ones(Float32,n)  # porosity
R(c::Vector{Matrix{Float32}}) = Patchy(c,1f3*vp,1f3*rho,phi; bulk_min = 5f10)[1]/1f3
vps = R(sw_pad)   # time-varying vp

##### Wave equation
o = (0f0, 0f0)          # origin

nsrc = 32       # num of sources
nrec = 600      # num of receivers

models = [Model(n, d, o, (1f0 ./ vps[i]).^2f0; nb = 80) for i = 1:nv]   # wave model

timeS = timeR = 3600f0               # recording time
dtS = dtR = 4f0                     # recording time sampling rate
ntS = Int(floor(timeS/dtS))+1       # time samples
ntR = Int(floor(timeR/dtR))+1       # source time samples
idx_wb = minimum(find_water_bottom(vp.-minimum(vp)))

extentx = (n[1]-1)*d[1];
extentz = (n[2]-1)*d[2];

mode = "transmission"
if mode == "reflection"
    xsrc = [convertToCell(Float32.(ContJitter(extentx, nsrc))) for i=1:nv]
    zsrc = [convertToCell(range(10f0,stop=10f0,length=nsrc)) for i=1:nv]
    xrec = range(d[1],stop=(n[1]-1)*d[1],length=nrec)
    zrec = range((idx_wb-1)*d[2]-2f0,stop=(idx_wb-1)*d[2]-2f0,length=nrec)
elseif mode == "transmission"
    xsrc = [convertToCell(range(d[1],stop=d[1],length=nsrc)) for i=1:nv]
    zsrc = [convertToCell(Float32.(ContJitter(extentz, nsrc))) for i=1:nv]
    xrec = range((n[1]-1)*d[1],stop=(n[1]-1)*d[1], length=nrec)
    zrec = range((idx_wb-1)*d[2]+10f0,stop=(n[2]-1)*d[2],length=nrec)
else
    # source locations -- half at the left hand side of the model, half on top
    xsrc = [convertToCell(vcat(Float32.(ContJitter(extentx, div(nsrc,2))),range(d[1],stop=d[1],length=div(nsrc,2)))) for i = 1:nv]
    zsrc = [convertToCell(vcat(range(10f0,stop=10f0,length=div(nsrc,2)),Float32.(ContJitter(extentz, div(nsrc,2))))) for i = 1:nv]
    xrec = vcat(range((n[1]-1)*d[1],stop=(n[1]-1)*d[1], length=div(nrec,2)),range(d[1],stop=(n[1]-1)*d[1],length=div(nrec,2)))
    zrec = vcat(range((idx_wb-1)*d[2]+10f0,stop=(n[2]-1)*d[2],length=div(nrec,2)),range(10f0,stop=10f0,length=div(nrec,2)))
end

ysrc = convertToCell(range(0f0,stop=0f0,length=nsrc))
yrec = 0f0

# set up src/rec geometry
srcGeometry = [Geometry(xsrc[i], ysrc, zsrc[i]; dt=dtS, t=timeS) for i = 1:nv]
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# set up source
f0 = 0.02f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = [judiVector(srcGeometry[i], wavelet) for i = 1:nv]

# set up simulation operators
Fs = [judiModeling(models[i], srcGeometry[i], recGeometry) for i = 1:nv] # acoustic wave equation solver

## wave physics
function F(v::Vector{Matrix{Float32}})
    m = [vec(1f0./v[i]).^2f0 for i = 1:nv]
    return [Fs[i](m[i], q[i]) for i = 1:nv]
end

# Define seismic data directory
mkpath(datadir("seismic-data"))
misc_dict = @strdict mode nsrc nrec nv f0 cut_area tstep factor d n kvoverkh

### generate/load data
if ~isfile(datadir("seismic-data", savename(misc_dict, "jld2"; digits=6)))
    println("generating data")
    global d_obs = [Fs[i]*q[i] for i = 1:nv]
    seismic_dict = @strdict mode nsrc nrec nv f0 cut_area tstep factor d n d_obs q srcGeometry recGeometry model kvoverkh
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

ls = BackTracking(order=3, iterations=10)

# Main loop
niterations = 200
nssample = 4
fhistory = zeros(niterations)

### inversion initialization
logK0 = deepcopy(logK)
logK0[v.>3.5] .= log(100. * md)
dlogK = 0 .* logK0
logK_init = deepcopy(logK0)
y_init = box_co2(O(S(T(logK_init), f)))

device = gpu
info = JSON.parsefile(projectdir("info.json"))

## load NF
net_path = datadir("trained-NF", info["_NF_NAME"])
mkpath(datadir("trained-NF"))

# Download the dataset into the data directory if it does not exist
if ~isfile(net_path)
        run(`wget $(info["_NF_LINK"])'
        '$(info["_NF_NAME"]) -q -O $net_path`)
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

normal_min = network_dict["normal_min"]
normal_max = network_dict["normal_max"]
@. normal(x; normal_min=normal_min, normal_max=normal_max) = (x-normal_min)/(normal_max-normal_min)
@. invnormal(x; normal_min=normal_min, normal_max=normal_max) = x*(normal_max-normal_min)+normal_min

# check generative samples are good so that loading went well. 
G.forward(randn(Float32,network_dict["nx"],network_dict["ny"],1,1));
gen = invnormal(G.inverse(randn(Float32,network_dict["nx"],network_dict["ny"],1,1)));

# generator now
G = G |> device;
G = reverse(G);

z = G.inverse(Float32.(normal(reshape(Float32.(logK0), ns[1], ns[end], 1, 1))) |> device)
λ = 4f0

for j=1:niterations

    Base.flush(Base.stdout)

    ### subsample sources
    rand_ns = [jitter(nsrc, nssample) for i = 1:nv]                             # select random source idx for each vintage
    q_sub = [q[i][rand_ns[i]] for i = 1:nv]                                        # set-up source
    F_sub = [Fs[i][rand_ns[i]] for i = 1:nv]                                 # set-up wave modeling operator
    dobs = [d_obs[i][rand_ns[i]] for i = 1:nv]                                  # subsampled seismic dataset from the selected sources
    function F(v::Vector{Matrix{Float32}})
        m = [vec(1f0./v[i]).^2f0 for i = 1:nv]
        return [F_sub[i](m[i], q_sub[i]) for i = 1:nv]
    end

    # objective function for inversion
    function obj(z)
        global logK_j = box_logK(invnormal(Float64.(G(z)|>cpu)[:,:,1,1]))
        global c_j = box_co2(M(O(S(T(logK_j), f))))
        global dpred_j = F(box_v(R(pad(c_j))))
        fval = .5f0 * norm(dpred_j-dobs)^2f0/nssample/nv + .5f0 * λ^2f0 * norm(z)^2f0/length(z) 
        @show fval
        return fval
    end
    ## AD by Flux
    @time fval, gs = Flux.withgradient(() -> obj(z), Flux.params(z))
    g = gs[z]
    p = -g

    # linesearch
    function f_(α)
        misfit = obj(Float32.(z + α * p))
        @show α, misfit
        return misfit
    end
    
    step, fval = ls(f_, 0.3f0, fval, dot(g, p))
    global z = Float32.(z + step * p)
        
    println("Inversion iteration no: ",j,"; function value: ",fval)
    fhistory[j] = fval
    
    ### save intermediate results
    save_dict = @strdict mode j λ z nssample f0 logK_j g step niterations nv nsrc nrec nv cut_area tstep factor n d fhistory mask kvoverkh
    @tagsave(
        joinpath(datadir(sim_name, exp_name), savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

    ## save figure
    fig_name = @strdict mode j nssample f0 z niterations nv nsrc nrec nv cut_area tstep factor n d fhistory mask kvoverkh

    ## compute true and plot
    SNR = -2f1 * log10(norm(K-exp.(logK_j))/norm(K))
    fig = figure(figsize=(20,12));
    subplot(2,2,1);
    imshow(exp.(logK_j)'./md, norm=matplotlib.colors.LogNorm(vmin=50, vmax=maximum(exp.(logK)./md)));title("inversion, $(j) iter");colorbar();
    subplot(2,2,2);
    imshow(K'./md, norm=matplotlib.colors.LogNorm(vmin=50, vmax=maximum(exp.(logK)./md)));title("GT permeability");colorbar();
    subplot(2,2,3);
    imshow(exp.(logK_init)'./md, norm=matplotlib.colors.LogNorm(vmin=50, vmax=maximum(exp.(logK)./md)));title("initial permeability");colorbar();
    subplot(2,2,4);
    imshow(abs.(K'-exp.(logK_j)')./md.+eps(), norm=matplotlib.colors.LogNorm(vmin=50, vmax=maximum(exp.(logK)./md)));title("abs error, SNR=$SNR");colorbar();
    suptitle("End-to-end Inversion at iter $(j)")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_K.png"), fig);
    close(fig)

    ## loss
    fig = figure(figsize=(20,12));
    plot(fhistory[1:j]);title("loss=$(fhistory[j])");
    suptitle("End-to-end Inversion at iter $(j)")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_loss.png"), fig);
    close(fig)

    ## data fitting
    fig = figure(figsize=(20,12));
    for i = 1:5
        subplot(4,5,i);
        imshow(y_init[i]', vmin=0, vmax=1);
        title("initial prediction at snapshot $(2*i)")
        subplot(4,5,i+5);
        imshow(sw_true[i]', vmin=0, vmax=1);
        title("true at snapshot $(i)")
        subplot(4,5,i+10);
        imshow(c_j[i]', vmin=0, vmax=1);
        title("predict at snapshot $(2*i)")
        subplot(4,5,i+15);
        imshow(5*(sw_true[i]'-c_j[i]'), vmin=-1, vmax=1, cmap="magma");
        title("5X diff at snapshot $(2*i)")
    end
    suptitle("End-to-end Inversion at iter $(j)")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_saturation.png"), fig);
    close(fig)

end
