## A 2D compass example

using DrWatson
@quickactivate "jutul-compass"

using Pkg; Pkg.instantiate()
using JutulDarcyRules
using LinearAlgebra
using PyPlot
using Flux
using LineSearches
using JLD2
using JUDI
using Statistics
using InvertibleNetworks
using GlowInvertibleNetwork
using Random
using Images
using FNO4CO2
using JSON

if gpu_flag
    device = gpu
else
    device = cpu
end

matplotlib.use("agg")
include(srcdir("dummy_src_file.jl"))

sim_name = "2D-K-inv"
exp_name = "compass-NF-FNO"

mkpath(datadir())
mkpath(plotsdir())

info = JSON.parsefile(projectdir("info.json"))

## load FNO
net_path_FNO = datadir("3D_FNO", info["_FNO_NAME"])
mkpath(datadir("3D_FNO"))

# Download the dataset into the data directory if it does not exist
if ~isfile(net_path_FNO)
        run(`wget $(info["_FNO_LINK"])'
        '$(info["_FNO_NAME"]) -q -O $net_path_FNO`)
end

net_dict_FNO = JLD2.jldopen(net_path_FNO, "r")
NN = net_dict_FNO["NN_save"];
AN = net_dict_FNO["AN"];
grid_ = gen_grid(net_dict_FNO["n"], net_dict_FNO["d"], net_dict_FNO["nt"], net_dict_FNO["dt"]);
Flux.testmode!(NN, true);

function S(x)
    return clamp.(NN(perm_to_tensor(x, grid_, AN)), 0f0, 0.9f0);
end

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

n = ns
d = ds

ϕ = 0.25
model = jutulModel(n, d, ϕ, K1to3(K; kvoverkh=0.36); h=h)

## simulation time steppings
tstep = 365.25 * ones(15)
tot_time = sum(tstep)

## injection & production
inj_loc = (Int(round(n[1]/2)), 1, n[end]-20) .* d
irate = 0.3
q = jutulForce(irate, [inj_loc])

## set up modeling operator
Sjutul = jutulModeling(model, tstep)

## simulation
mesh = CartesianMesh(model)
T(x) = log.(KtoTrans(mesh, K1to3(exp.(x); kvoverkh=0.36)))

logK = log.(K)

seal_mask = Kh.==1e-3
set_seal(x::AbstractMatrix{T}) where T = x .* convert(typeof(x), (T(1) .- T.(seal_mask))) + convert(typeof(x), T(log(1e-3*md)) * T.(seal_mask))

# Download the dataset into the data directory if it does not exist
mkpath(datadir("flow-data"))
if ~isfile(datadir("flow-data", "true_state.jld2"))
    run(`wget https://www.dropbox.com/s/ts2wntxnqy1zqdr/'
        'true_state.jld2 -q -O $(datadir("flow-data", "true_state.jld2"))`)
end
JLD2.@load datadir("flow-data", "true_state.jld2") state

#### inversion
ls = BackTracking(order=3, iterations=10)

# Main loop
niterations = 500
fhistory = zeros(niterations)

function VtoK_nowater(v::Matrix{T}, d::Tuple{T, T}; α::T=T(20)) where T

    n = size(v)
    idx_ucfmt = find_water_bottom((v.-T(3.5)).*(v.>T(3.5)))

    return vcat([vcat(
        α * exp.(v[i,1:idx_ucfmt[i]-1]) .- α*exp(T(1.48)),
        α*exp.(v[i,idx_ucfmt[i]:end])  .- α*exp(T(3.5)))' for i = 1:n[1]]...)
end

logK = Float32.(logK)
logK0 = log.(VtoK_nowater(v, (d[1], d[end])).*md)
logK0[v.>3.5] .= mean(logK[v.>3.5])
logK0 = max.(logK0, log(20*md))

z = G.inverse(normal(reshape(Float32.(logK0), ns[1], ns[end], 1, 1)) |> device)
#z = 0f0 .* z
#z = randn(Float32, ns[1], ns[end], 1, 1) |> device
λ = 5f0

# GD algorithm
learning_rate = 1f-1
lr_min = learning_rate*1f-2
decay_rate = exp(log(lr_min/learning_rate)/niterations)
opt = Flux.Optimiser(ExpDecay(learning_rate, decay_rate, 1, lr_min), Descent(1f0))

ctrue = state[1:length(tstep)*prod(n)] |> device
NN = NN |> device;
AN = AN |> device;
z = z |> device;
#init_misfit = norm(vec(S(box_logK(set_seal(invnormal(G(z)[:,:,1,1]))))[:,:,1:15,1])-ctrue)^2
#noise = vec(S(Float32.(logK)|>device)[:,:,1:15,1])-ctrue
#σ = Float32.(norm(noise)/sqrt(length(noise)))
#f(z) = .5 * norm(vec(S(box_logK(set_seal(invnormal(G(z)[:,:,1,1]))))[:,:,1:15,1])-ctrue)^2/init_misfit + .5f0 * λ^2f0 * norm(z)^2f0/length(z) 
f(z) = .5f0 * norm(vec(S(box_logK(set_seal(invnormal(G(z)[:,:,1,1]))))[:,:,1:15,1])-ctrue)^2 + .5f0 * λ^2f0 * norm(z)^2f0
logK_init = box_logK(set_seal(invnormal(G(z)[:,:,1,1])))
@time state_init = S(logK_init)

ls = BackTracking(c_1=1f-4,iterations=10,maxstep=Inf32,order=3,ρ_hi=5f-1,ρ_lo=1f-1)

for j=1:niterations

    @time fval, gs = Flux.withgradient(() -> f(z), Flux.params(z))

    # (normalized) update direction
    g = gs[z]
    p = -g

    # linesearch
    function ϕ_(α)::Float32
        fval = f(Float32.(z .+ α * p))
        isnan(fval) && return Inf32
        @show fval
        return fval
    end

    global step, fval = ls(ϕ_, 1f-1, fval, dot(g, p))
    global z = Float32.(z .+ step * p)
    fhistory[j] = fval
    
    println("Inversion iteration no: ",j,"; function value: ",fval)

    fig_name = @strdict j n d ϕ z tstep irate niterations inj_loc λ

    logK0 = box_logK(set_seal(invnormal(G(z)[:,:,1,1])))

    ### plotting
    fig=figure(figsize=(20,12));
    subplot(1,3,1);
    imshow(exp.(logK)'./md, norm=matplotlib.colors.LogNorm(vmin=200, vmax=maximum(exp.(logK)./md))); colorbar(); title("true permeability")
    subplot(1,3,2);
    imshow(exp.(logK0|>cpu)'./md, norm=matplotlib.colors.LogNorm(vmin=200, vmax=maximum(exp.(logK)./md))); colorbar(); title("inverted permeability")
    subplot(1,3,3);
    imshow(abs.(exp.(logK)'./md.-exp.(logK0|>cpu)'./md).+eps(), norm=matplotlib.colors.LogNorm(vmin=200, vmax=maximum(exp.(logK)./md))); colorbar(); title("diff")
    suptitle("Flow Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_diff.png"), fig);
    close(fig)

    state_predict = S(logK0) |> cpu

    ## data fitting
    fig = figure(figsize=(20,12));
    for i = 1:5
        subplot(4,5,i);
        imshow(state_init[:,:,3*i,1]', vmin=0, vmax=0.9); colorbar();
        title("initial prediction at snapshot $(3*i)")
        subplot(4,5,i+5);
        imshow(reshape(Saturations(state.states[3*i]), n[1], n[end])', vmin=0, vmax=0.9); colorbar();
        title("true at snapshot $(3*i)")
        subplot(4,5,i+10);
        imshow(state_predict[:,:,3*i,1]', vmin=0, vmax=0.9); colorbar();
        title("predict at snapshot $(3*i)")
        subplot(4,5,i+15);
        imshow(5*abs.(reshape(Saturations(state.states[3*i]), n[1], n[end])'-state_predict[:,:,3*i,1]'), vmin=0, vmax=0.9); colorbar();
        title("5X diff at snapshot $(3*i)")
    end
    suptitle("Flow Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_co2.png"), fig);
    close(fig)

    ## loss
    fig = figure(figsize=(20,12));
    plot(fhistory[1:j]);title("loss=$(fhistory[j])");
    suptitle("Flow Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_loss.png"), fig);
    close(fig)

end
