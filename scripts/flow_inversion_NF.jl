## A 2D compass example

using DrWatson
@quickactivate "jutul-compass"

using Pkg; Pkg.instantiate()

nthreads = try
    # Slurm
    parse(Int, ENV["SLURM_CPUS_ON_NODE"])
catch e
    # Desktop
    Sys.CPU_THREADS
end
using LinearAlgebra
BLAS.set_num_threads(nthreads)

using JutulDarcyRules
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
using JSON

device = gpu

matplotlib.use("agg")
include(srcdir("dummy_src_file.jl"))

sim_name = "2D-K-inv"
exp_name = "compass-NF"

mkpath(datadir())
mkpath(plotsdir())

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

n = ns
d = ds

ϕ = 0.25
model = jutulModel(n, d, ϕ, K1to3(K; kvoverkh=0.1); h=h)

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
T(x) = log.(KtoTrans(mesh, K1to3(exp.(x); kvoverkh=0.1)))

logK = log.(K)

@time state = S(T(logK), f)

# Main loop
niterations = 500
fhistory = zeros(niterations)

### inversion initialization
logK0 = deepcopy(logK)
logK0[v.>3.5] .= mean(logK[v.>3.5])
logK_init = deepcopy(logK0)

z = G.inverse(Float32.(normal(reshape(Float32.(logK0), ns[1], ns[end], 1, 1))) |> device)
λ = 1f0

ctrue = state[1:length(tstep)*prod(n)]
init_misfit = norm(S(T(box_logK(invnormal(Float64.(G(z)|>cpu)[:,:,1,1]))), f)[1:length(tstep)*prod(n)]-ctrue)^2
function obj(z)
    global logK_j = box_logK(invnormal(Float64.(G(z)|>cpu)[:,:,1,1]))
    global c_j = S(T(logK_j), f)
    return .5 * norm(c_j[1:length(tstep)*prod(n)]-ctrue)^2/init_misfit + .5f0 * λ^2f0 * norm(z)^2f0/length(z) 
end

logK_init = box_logK(invnormal(Float64.(G(z)|>cpu)[:,:,1,1]))
@time state_init = S(T(logK_init), f)

ls = BackTracking(order=3, iterations=10)

for j=1:niterations

    @time fval, gs = Flux.withgradient(() -> obj(z), Flux.params(z))
    fhistory[j] = fval
    g = gs[z]
    p = -g

    # linesearch
    function f_(α)
        misfit = obj(Float32.(z + α * p))
        @show α, misfit
        return misfit
    end

    step, fval = ls(f_, 100.0, fval, dot(g, p))
    global z = Float32.(z + step * p)
    
    println("Inversion iteration no: ",j,"; function value: ",fval)

    fig_name = @strdict j n d ϕ z tstep irate niterations inj_loc λ

    ### plotting
    fig=figure(figsize=(20,12));
    subplot(1,3,1);
    imshow(exp.(logK)'./md, norm=matplotlib.colors.LogNorm(vmin=200, vmax=maximum(exp.(logK)./md))); colorbar(); title("true permeability")
    subplot(1,3,2);
    imshow(exp.(logK_j)'./md, norm=matplotlib.colors.LogNorm(vmin=200, vmax=maximum(exp.(logK)./md))); colorbar(); title("inverted permeability")
    subplot(1,3,3);
    imshow(abs.(exp.(logK)'./md.-exp.(logK_j)'./md), norm=matplotlib.colors.LogNorm(vmin=200, vmax=maximum(exp.(logK)./md))); colorbar(); title("diff")
    suptitle("Flow Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_diff.png"), fig);
    close(fig)

    ## data fitting
    fig = figure(figsize=(20,12));
    for i = 1:5
        subplot(4,5,i);
        imshow(reshape(Saturations(state_init.states[i]), n[1], n[end])', vmin=0, vmax=0.9); colorbar();
        title("initial prediction at snapshot $(i)")
        subplot(4,5,i+5);
        imshow(reshape(Saturations(state.states[i]), n[1], n[end])', vmin=0, vmax=0.9); colorbar();
        title("true at snapshot $(i)")
        subplot(4,5,i+10);
        imshow(reshape(Saturations(c_j.states[i]), n[1], n[end])', vmin=0, vmax=0.9); colorbar();
        title("predict at snapshot $(i)")
        subplot(4,5,i+15);
        imshow(5*abs.(reshape(Saturations(state.states[i]), n[1], n[end])'-reshape(Saturations(c_j.states[i]), n[1], n[end])'), vmin=0, vmax=0.9); colorbar();
        title("5X diff at snapshot $(i)")
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
