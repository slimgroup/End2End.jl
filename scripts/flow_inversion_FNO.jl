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
using FNO4CO2

matplotlib.use("agg")
include(srcdir("dummy_src_file.jl"))

sim_name = "2D-K-inv"
exp_name = "compass-FNO"

mkpath(datadir())
mkpath(plotsdir())

## load FNO
net_path_FNO = datadir("3D_FNO", "batch_size=8_dt=0.058824_ep=70_epochs=1000_learning_rate=0.0001_modet=9_modex=16_modez=16_nt=18_ntrain=5960_nvalid=40_s=1_width=20.jld2")
mkpath(datadir("3D_FNO"))

# Download the dataset into the data directory if it does not exist
if ~isfile(net_path_FNO)
        run(`wget https://www.dropbox.com/s/3q3g6ixzavaikdb/'
        'batch_size=8_dt=0.058824_ep=70_epochs=1000_learning_rate=0.0001_modet=9_modex=16_modez=16_nt=18_ntrain=5960_nvalid=40_s=1_width=20.jld2 -q -O $net_path_FNO`)
end

net_dict_FNO = JLD2.jldopen(net_path_FNO, "r")
NN = net_dict_FNO["NN_save"];
AN = net_dict_FNO["AN"];
grid_ = gen_grid(net_dict_FNO["n"], net_dict_FNO["d"], net_dict_FNO["nt"], net_dict_FNO["dt"]);
Flux.testmode!(NN, true);

function S(x)
    return clamp.(NN(perm_to_tensor(x, grid_, AN)), 0f0, 0.9f0);
end

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

logK = log.(K)

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

@time state = Sjutul(T(logK), q);
@time state_FNO = S(Float32.(logK));

#### inversion
logK0 = deepcopy(logK)
logK0[v.>3.5] .= mean(logK[v.>3.5])
logK0 = Float32.(logK0)
logK_init = deepcopy(logK0)

@time state_init = S(Float32.(logK_init));

ctrue = state[1:length(tstep)*prod(n)]
prj = box_logK
f(logK) = .5 * norm(vec(S(prj(logK))[:,:,1:15,1])-ctrue)^2

# Main loop
niterations = 100
fhistory = zeros(Float32,niterations)

# GD algorithm
learning_rate = 0.02f0
lr_min = learning_rate*1f-2
decay_rate = exp(log(lr_min/learning_rate)/niterations)
opt = Flux.Optimiser(ExpDecay(learning_rate, decay_rate, 1, lr_min), Descent(1f0))

for j=1:niterations

    @time fval, gs = Flux.withgradient(() -> f(logK0), Flux.params(logK0));
    g = gs[logK0]
    fhistory[j] = fval
    Flux.Optimise.update!(opt, logK0, g)

    fig_name = @strdict j n d ϕ logK0 tstep irate niterations inj_loc

    ### plotting
    fig=figure(figsize=(20,12));
    subplot(1,3,1);
    imshow(exp.(logK)'./md, norm=matplotlib.colors.LogNorm(vmin=200, vmax=maximum(exp.(logK)./md))); colorbar(); title("true permeability")
    subplot(1,3,2);
    imshow(exp.(prj(logK0))'./md, norm=matplotlib.colors.LogNorm(vmin=200, vmax=maximum(exp.(logK)./md))); colorbar(); title("inverted permeability")
    subplot(1,3,3);
    imshow(abs.(exp.(logK)'./md.-exp.(prj(logK0))'./md), norm=matplotlib.colors.LogNorm(vmin=200, vmax=maximum(exp.(logK)./md))); colorbar(); title("diff")
    suptitle("Flow Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_diff.png"), fig);
    close(fig)

    state_predict = S(logK0)

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
