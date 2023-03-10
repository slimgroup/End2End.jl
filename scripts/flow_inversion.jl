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

matplotlib.use("agg")
include(srcdir("dummy_src_file.jl"))

sim_name = "2D-K-inv"
exp_name = "compass"

mkpath(datadir())
mkpath(plotsdir())

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
logK0 = deepcopy(logK)
logK0[v.>3.5] .= mean(logK[v.>3.5])
logK_init = deepcopy(logK0)

state_init = S(T(logK_init), q)

f(logK) = .5 * norm(S(T(logK),q)[1:length(tstep)*prod(n)]-state[1:length(tstep)*prod(n)])^2
ls = BackTracking(order=3, iterations=10)

lower, upper = 1.1*minimum(logK), 0.9*maximum(logK)
prj(x) = max.(min.(x,upper),lower)
# Main loop
niterations = 100
fhistory = zeros(niterations)

for j=1:niterations

    @time fval, gs = Flux.withgradient(() -> f(logK0), Flux.params(logK0))
    g = gs[logK0]
    p = -g/norm(g, Inf)
    
    println("Inversion iteration no: ",j,"; function value: ",fval)
    fhistory[j] = fval

    # linesearch
    function f_(α)
        misfit = f(prj(logK0 .+ α * p))
        @show α, misfit
        return misfit
    end

    step, fval = ls(f_, 1e-1, fval, dot(g, p))

    # Update model and bound projection
    global logK0 = prj(logK0 .+ step .* p)

    fig_name = @strdict j n d ϕ logK0 tstep irate niterations lower upper inj_loc

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