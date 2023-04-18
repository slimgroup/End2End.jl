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
using Statistics
using InvertibleNetworks
using GlowInvertibleNetwork
using Random
using Images
using JSON
using Polynomials

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

# check generative samples are good so that loading went well. 
G.forward(randn(Float32,network_dict["nx"],network_dict["ny"],1,1));
gen = G.inverse(randn(Float32,network_dict["nx"],network_dict["ny"],1,1));

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
ds = (ds[1], ds[1]*ns[1]/5, ds[2])

Kh = VtoK.(v);
K = Float64.(Kh * md);

# set up jutul model
kvoverkh = 0.1
α = 6.0
ϕ = Ktoϕ.(Kh; α=α)
model = jutulModel(ns, ds, vec(padϕ(ϕ)), K1to3(K; kvoverkh=kvoverkh), h)

## simulation time steppings
tstep = 365.25 * 12 * ones(5)
tot_time = sum(tstep)

## injection & production
inj_loc = (128, 1) .* ds[1:2]
startz = (ns[end]-18) * ds[end]
endz = (ns[end]-16) * ds[end]
pore_volumes = sum(ϕ[2:end-1,1:end-1] .* (v[2:end-1,1:end-1].>3.5)) * prod(ds)
irate = 0.2 * pore_volumes / tot_time / 24 / 60 / 60

f = jutulVWell(irate, inj_loc; startz = startz, endz = endz)

## set up modeling operator
S = jutulModeling(model, tstep)

## simulation
mesh = CartesianMesh(model)
T(x) = log.(KtoTrans(mesh, K1to3(exp.(x); kvoverkh=kvoverkh)))

logK = log.(K)

F(ϕ) = S(T(log.(ϕtoK.(ϕ;α=α)*md)),vec(padϕ(ϕ)),f)
@time state = F(ϕ)
# Main loop
niterations = 500
fhistory = zeros(niterations)

#### inversion
ϕ0 = deepcopy(ϕ)
ϕ_init_val = mean(ϕ0[v.>3.5])
ϕ0[v.>3.5] .= ϕ_init_val
ϕ0_init = deepcopy(ϕ0)
dϕ = 0 .* ϕ
ϕ_init = deepcopy(ϕ0)

logK0 = log.(ϕtoK.(ϕ0;α=α)*md)
logK_init = deepcopy(logK0)
@time y_init = F(ϕ0)

z = G.inverse(Float32.(reshape(ϕ0, ns[1], ns[end], 1, 1)) |> device)
z = 0.8f0 * z/norm(z) * Float32(sqrt(length(z)))
λ = 0f0

ctrue = state[1:length(tstep)*prod(ns)]
init_misfit = norm(F(ϕ0)[1:length(tstep)*prod(ns)]-ctrue)^2

function obj(z)
    global ϕ_j = box_ϕ(Float64.(G(z)|>cpu)[:,:,1,1])
    global c_j = F(ϕ_j)
    return .5 * norm(c_j[1:length(tstep)*prod(ns)]-ctrue)^2 + .5f0 * λ^2f0 * norm(z)^2f0/length(z) 
end

ϕ_init = box_ϕ(Float64.(G(z)|>cpu)[:,:,1,1])
@time state_init = F(ϕ_init)

ls = BackTracking(order=3, iterations=10)

for j=1:niterations

    @time fval, gs = Flux.withgradient(() -> obj(z), Flux.params(z))
    fhistory[j] = fval
    g = gs[z]
    p = -g

    # linesearch
    function f_(α)
        try
            misfit = obj(Float32.(z + α * p))
            @show α, misfit
            return misfit
        catch e
            return Inf
        end
    end

    step, fval = ls(f_, 2f-2, fval, dot(g, p))
    global z = Float32.(z + step * p)
    
    println("Inversion iteration no: ",j,"; function value: ",fval)

    fig_name = @strdict α ϕ_init_val j n d ϕ z tstep irate niterations inj_loc λ

    ## compute true and plot
    SNR = -2f1 * log10(norm(ϕ-ϕ_j)/norm(ϕ))
    fig = figure(figsize=(20,12));
    subplot(2,2,1);
    imshow(ϕ_j', vmin=0, vmax=maximum(ϕ));title("inversion, SNR=$(SNR)");colorbar();
    subplot(2,2,2);
    imshow(ϕ', vmin=0, vmax=maximum(ϕ));title("GT permeability");colorbar();
    subplot(2,2,3);
    imshow(ϕ_init', vmin=0, vmax=maximum(ϕ));title("initial permeability");colorbar();
    subplot(2,2,4);
    imshow(ϕ_j'-ϕ_init', vmin=-0.5*maximum(ϕ), vmax=0.5*maximum(ϕ), cmap="magma");title("updated");colorbar();
    suptitle("Inversion at iter $(j)")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_ϕ.png"), fig);
    close(fig)
 
    ## data fitting
    fig = figure(figsize=(20,12));
    for i = 1:5
        subplot(4,5,i);
        imshow(reshape(Saturations(state_init.states[i]), ns[1], ns[end])', vmin=0, vmax=0.9); colorbar();
        title("initial prediction at snapshot $(i)")
        subplot(4,5,i+5);
        imshow(reshape(Saturations(state.states[i]), ns[1], ns[end])', vmin=0, vmax=0.9); colorbar();
        title("true at snapshot $(i)")
        subplot(4,5,i+10);
        imshow(reshape(Saturations(c_j.states[i]), ns[1], ns[end])', vmin=0, vmax=0.9); colorbar();
        title("predict at snapshot $(i)")
        subplot(4,5,i+15);
        imshow(5*abs.(reshape(Saturations(state.states[i]), ns[1], ns[end])'-reshape(Saturations(c_j.states[i]), ns[1], ns[end])'), vmin=0, vmax=0.9); colorbar();
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
