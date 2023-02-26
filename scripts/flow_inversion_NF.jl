## A 2D compass example

using DrWatson
@quickactivate "jutul-compass"

using Pkg; Pkg.instantiate()
using JutulDarcyAD
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

matplotlib.use("agg")
include(srcdir("dummy_src_file.jl"))

sim_name = "2D-K-inv"
exp_name = "compass-NF"

mkpath(datadir())
mkpath(plotsdir())

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
S = jutulModeling(model, tstep)

## simulation
mesh = CartesianMesh(model)
T(x) = log.(KtoTrans(mesh, K1to3(exp.(x); kvoverkh=0.36)))

logK = log.(K)

@time state = S(T(logK), q)

#### inversion
ls = BackTracking(order=3, iterations=10)

# Main loop
niterations = 100
fhistory = zeros(niterations)

logK0 = deepcopy(logK)
logK0[v.>3.5] .= mean(logK[v.>3.5])
z = G.inverse(reshape(Float32.(logK0), ns[1], ns[end], 1, 1))
λ = 1f0
# ADAM-W algorithm
lower, upper = log(1e-4*md), log(1500*md)
box_logK(x::AbstractArray{T}) where T = max.(min.(x,T(upper)),T(lower))

# ADAM-W algorithm
learning_rate = 1f-1
opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4)

niterations = 100
init_misfit = norm(S(T(box_logK(Float64.(G(z)[:,:,1,1]))),q)[1:length(tstep)*prod(n)]-state[1:length(tstep)*prod(n)])^2
f(z) = .5 * norm(S(T(box_logK(Float64.(G(z)[:,:,1,1]))),q)[1:length(tstep)*prod(n)]-state[1:length(tstep)*prod(n)])^2/init_misfit + .5f0 * λ^2f0 * norm(z)^2f0/length(z) 

logK_init = box_logK(Float64.(G(z)[:,:,1,1]))
@time state_init = S(T(logK_init), q)

for j=1:niterations

    @time fval, gs = Flux.withgradient(() -> f(z), Flux.params(z))
    Flux.Optimise.update!(opt, z, gs[z])
    fhistory[j] = fval
    
    println("Inversion iteration no: ",j,"; function value: ",fval)

    box_logK(Float64.(G(z)[:,:,1,1]))

    fig_name = @strdict j n d ϕ z tstep irate niterations lower upper inj_loc λ

    logK0 = box_logK(Float64.(G(z)[:,:,1,1]))
    state_predict = S(T(logK0), q)

    ### plotting
    fig=figure(figsize=(20,12));
    subplot(1,3,1);
    imshow(exp.(logK)'./md, vmin=minimum(exp.(logK))./md, vmax=maximum(exp.(logK)./md)); colorbar(); title("true permeability")
    subplot(1,3,2);
    imshow(exp.(logK0)'./md, vmin=minimum(exp.(logK))./md, vmax=maximum(exp.(logK)./md)); colorbar(); title("inverted permeability")
    subplot(1,3,3);
    imshow(exp.(logK)'./md.-exp.(logK0)'./md, vmin=minimum(exp.(logK)), vmax=maximum(exp.(logK)./md)); colorbar(); title("diff")
    suptitle("Flow Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_diff.png"), fig);
    close(fig)

    state_predict = S(T(logK0), q)

    ## data fitting
    fig = figure(figsize=(20,12));
    for i = 1:5
        subplot(4,5,i);
        imshow(reshape(Saturations(state_init.states[3*i]), n[1], n[end])', vmin=0, vmax=0.9); colorbar();
        title("initial prediction at snapshot $(3*i)")
        subplot(4,5,i+5);
        imshow(reshape(Saturations(state.states[3*i]), n[1], n[end])', vmin=0, vmax=0.9); colorbar();
        title("true at snapshot $(3*i)")
        subplot(4,5,i+10);
        imshow(reshape(Saturations(state_predict.states[3*i]), n[1], n[end])', vmin=0, vmax=0.9); colorbar();
        title("predict at snapshot $(3*i)")
        subplot(4,5,i+15);
        imshow(5*abs.(reshape(Saturations(state.states[3*i]), n[1], n[end])'-reshape(Saturations(state_predict.states[3*i]), n[1], n[end])'), vmin=0, vmax=0.9); colorbar();
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
