## A 2D compass example

using DrWatson
@quickactivate "jutul-compass"

using Pkg; Pkg.instantiate();
using JutulDarcyRules
using LinearAlgebra
using PyPlot
using Flux
using LineSearches
using JLD2
using Statistics
using Polynomials

include(srcdir("dummy_src_file.jl"))
matplotlib.use("agg")

sim_name = "flow-inv"
exp_name = "SEG-compass"

mkpath(datadir())
mkpath(plotsdir())

## load compass
JLD2.@load datadir("image2023_v_rho.jld2") v rho
v = Float64.(v[:,182:end])
factor = (3,2)
d = (6., 6.)
h = 181 * d[end]

function downsample(v::Matrix{T}, factor::Tuple{Int, Int}) where T
    v_out_size = div.(size(v), factor)
    v_out = zeros(T, v_out_size)
    for i = 1:v_out_size[1]
        for j = 1:v_out_size[2]
            v_out[i,j] = mean(v[factor[1]*i-factor[1]+1:factor[1]*i, factor[2]*j-factor[2]+1:factor[2]*j])
        end
    end
    return v_out
end
v = 1.0./downsample(1.0./v, factor)
d = Float64.(d) .* factor;
function VtoK(v)
    if v >= 4
        K = exp(v-4.25) * 3000. / exp(4-4.25)
    elseif v >= 3.5
        K = 0.01 * exp(log(3000. /0.01)/0.5*(v-3.5))
    else
        K = 0.01 * exp(v-1.5)/exp(3.5-1.5)
    end
    return K
end

Kh = VtoK.(v);
#=
figure();imshow(Kh');colorbar();
fig = figure("pyplot_histogram",figsize=(10,10)) # Not strictly required
ax = PyPlot.axes() # Not strictly required
histogram = plt.hist(log.(vec(Kh[v.>3.5])*md), 20) # Histogram
grid("on")
xlabel("X")
ylabel("Y")
title("Histogram log K")
gcf()
savefig("histlogK.png", bbox_inches="tight", dpi=300)

fig = figure("pyplot_histogram",figsize=(10,10)) # Not strictly required
ax = PyPlot.axes() # Not strictly required
histogram = plt.hist((vec(v[v.>3.5])*md), 20) # Histogram

grid("on")
xlabel("X")
ylabel("Y")
title("Histogram v")
gcf()
=#
K = Float64.(Kh * md);
n = (size(K,1), 1, size(K,2))
d = (d[1], d[1]*n[1]/5, d[2])

α = 6.0
ϕ = Ktoϕ.(Kh; α=α)
kvoverkh = 0.1
model = jutulModel(n, d, vec(padϕ(ϕ)), K1to3(K; kvoverkh=kvoverkh), h)

## simulation time steppings
tstep = 365.25 * 12 * ones(5)
tot_time = sum(tstep)

## injection & production
inj_loc = (128, 1, n[end]-20) .* d
pore_volumes = sum(ϕ[2:end-1,1:end-1] .* (v[2:end-1,1:end-1].>3.5)) * prod(d)
irate = 0.2 * pore_volumes / tot_time / 24 / 60 / 60
#irate = 0.889681478439425
#irate = 0.3
q = jutulVWell(irate, (inj_loc[1], inj_loc[2]); startz = (n[end]-18) * d[end], endz = (n[end]-16) * d[end])

## set up modeling operator
S = jutulModeling(model, tstep)

## simulation
mesh_ = CartesianMesh(model)
T(x) = log.(KtoTrans(mesh_, K1to3(exp.(x); kvoverkh=kvoverkh)))

logK = log.(K)

@time state = S(T(logK), vec(padϕ(ϕ)), q)

#### inversion
ϕ0 = deepcopy(ϕ)
ϕ0[v.>3.5] .= 0.12
ϕ0_init = deepcopy(ϕ0)

logK0 = log.(ϕtoK.(ϕ0;α=α)*md)
logK_init = deepcopy(logK0)

@time state_init = S(T(logK_init), vec(padϕ(ϕ0)), q)

extent = (0f0, (n[1]-1)*d[1], (n[end]-1)*d[end]+h, 0f0+h)
figure(figsize=(10,6))
subplot(2,2,1);
scatter(range(q.loc[1][1], stop=q.loc[1][1], length=3), h .+ range(q.startz, stop=q.endz, length=3), label="injection well", marker="x", color="black", s=30);
legend()
imshow(exp.(logK)'./md, norm=matplotlib.colors.LogNorm(vmin=100, vmax=maximum(exp.(logK)./md)), extent=extent, aspect="auto"); xlabel("X [m]"); ylabel("Z [m]"); colorbar(); title("true permeability");
subplot(2,2,2);
scatter(range(q.loc[1][1], stop=q.loc[1][1], length=3), h .+ range(q.startz, stop=q.endz, length=3), label="injection well", marker="x", color="black", s=30);
legend()
imshow(exp.(logK0)'./md, norm=matplotlib.colors.LogNorm(vmin=100, vmax=maximum(exp.(logK)./md)), extent=extent, aspect="auto"); xlabel("X [m]"); ylabel("Z [m]"); colorbar(); title("init permeability");
subplot(2,2,3);
scatter(range(q.loc[1][1], stop=q.loc[1][1], length=3), h .+ range(q.startz, stop=q.endz, length=3), label="injection well", marker="x", color="black", s=30);
legend()
imshow(reshape(Saturations(state.states[end]), n[1], n[end])', extent=extent, aspect="auto", cmap="gnuplot"); xlabel("X [m]"); ylabel("Z [m]"); colorbar();title("true saturation after $(tot_time/365.25) years");
subplot(2,2,4);
scatter(range(q.loc[1][1], stop=q.loc[1][1], length=3), h .+ range(q.startz, stop=q.endz, length=3), label="injection well", marker="x", color="black", s=30);
legend()
imshow(reshape(Saturations(state_init.states[end]), n[1], n[end])', extent=extent, aspect="auto", cmap="gnuplot"); xlabel("X [m]"); ylabel("Z [m]"); colorbar();title("init saturation after $(tot_time/365.25) years");
tight_layout()
savefig("sat.png", bbox_inches="tight", dpi=300)

figure(figsize=(10,6))
subplot(2,2,1);
scatter(range(q.loc[1][1], stop=q.loc[1][1], length=3), h .+ range(q.startz, stop=q.endz, length=3), label="injection well", marker="x", color="black", s=30);
legend()
imshow(exp.(logK)'./md, norm=matplotlib.colors.LogNorm(vmin=100, vmax=maximum(exp.(logK)./md)), extent=extent, aspect="auto"); xlabel("X [m]"); ylabel("Z [m]"); colorbar(); title("true permeability");
subplot(2,2,2);
scatter(range(q.loc[1][1], stop=q.loc[1][1], length=3), h .+ range(q.startz, stop=q.endz, length=3), label="injection well", marker="x", color="black", s=30);
legend()
imshow(exp.(logK0)'./md, norm=matplotlib.colors.LogNorm(vmin=100, vmax=maximum(exp.(logK)./md)), extent=extent, aspect="auto"); xlabel("X [m]"); ylabel("Z [m]"); colorbar(); title("init permeability");
subplot(2,2,3);
scatter(range(q.loc[1][1], stop=q.loc[1][1], length=3), h .+ range(q.startz, stop=q.endz, length=3), label="injection well", marker="x", color="black", s=30);
legend()
imshow(reshape(Pressure(state.states[end]), n[1], n[end])', extent=extent, aspect="auto"); xlabel("X [m]"); ylabel("Z [m]"); colorbar();title("true pressure after $(tot_time/365.25) years");
subplot(2,2,4);
scatter(range(q.loc[1][1], stop=q.loc[1][1], length=3), h .+ range(q.startz, stop=q.endz, length=3), label="injection well", marker="x", color="black", s=30);
legend()
imshow(reshape(Pressure(state_init.states[end]), n[1], n[end])', extent=extent, aspect="auto"); xlabel("X [m]"); ylabel("Z [m]"); colorbar();title("init pressure after $(tot_time/365.25) years");
tight_layout()
savefig("p.png", bbox_inches="tight", dpi=300)

weight = repeat(vec(ϕ), length(tstep))
f(ϕ) = .5 * norm(S(T(log.(ϕtoK.(ϕ;α=α)*md)),vec(padϕ(ϕ)),q)[1:length(tstep)*prod(n)].*weight-state[1:length(tstep)*prod(n)].*weight)^2
ls = BackTracking(order=3, iterations=10)

lower, upper = 0, 1
prj(x) = max.(min.(x,upper),lower)
# Main loop
niterations = 100
fhistory = zeros(niterations)

for j=1:niterations

    @time fval, gs = Flux.withgradient(() -> f(ϕ0), Flux.params(ϕ0))
    g = gs[ϕ0]
    p = -g/norm(g, Inf)
    
    println("Inversion iteration no: ",j,"; function value: ",fval)
    fhistory[j] = fval

    # linesearch
    function f_(α)
        try
            misfit = f(prj(ϕ0 .+ α * p))
            @show α, misfit
            return misfit
        catch
            return Inf32
        end
        return Inf32
    end

    step, fval = ls(f_, 2e-1, fval, dot(g, p))

    # Update model and bound projection
    global ϕ0 = prj(ϕ0 .+ step .* p)

    fig_name = @strdict j n d ϕ0 tstep irate niterations lower upper inj_loc

    ### plotting
    fig=figure(figsize=(20,12));
    subplot(1,3,1);
    imshow(ϕ', vmin=0, vmax=1); colorbar(); title("true porosity")
    subplot(1,3,2);
    imshow(ϕ0', vmin=0, vmax=1); colorbar(); title("inverted porosity")
    subplot(1,3,3);
    imshow(abs.(ϕ'-ϕ0'), vmin=0, vmax=1); colorbar(); title("abs diff")
    suptitle("Flow Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_diff.png"), fig);
    close(fig)

    state_predict = S(T(log.(ϕtoK.(ϕ0;α=α)*md)),vec(padϕ(ϕ0)),q)

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
        imshow(reshape(Saturations(state_predict.states[i]), n[1], n[end])', vmin=0, vmax=0.9); colorbar();
        title("predict at snapshot $(i)")
        subplot(4,5,i+15);
        imshow(5*abs.(reshape(Saturations(state.states[i]), n[1], n[end])'-reshape(Saturations(state_predict.states[i]), n[1], n[end])'), vmin=0, vmax=0.9); colorbar();
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