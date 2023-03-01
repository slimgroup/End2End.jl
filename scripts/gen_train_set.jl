## A 2D compass example

using DrWatson
@quickactivate "jutul-compass"

using Pkg; Pkg.instantiate();
using JutulDarcyAD
using LinearAlgebra
using PyPlot
using Flux
using LineSearches
using JLD2
using JUDI
using Statistics
using Images
using Random
Random.seed!(2023)

include(srcdir("dummy_src_file.jl"))

sim_name = "gen_train"
exp_name = "compass"

mkpath(datadir())
mkpath(plotsdir())

## load compass
if ~isfile(datadir("v25-6.jld2"))
    run(`wget https://www.dropbox.com/s/mglbf450upcd768/'
        'v25-6.jld2 -q -O $(datadir("v25-6.jld2"))`)
end
JLD2.@load datadir("v25-6.jld2") v;
v6 = v./1f3;

# downsample
nsample = 6000
factor = 2
n = (128, 1, 80)
d = (12., 2000., 12.)
h = 181 * 6.

## slice locations
rand_x_start_list = rand(1:size(v6, 1)-factor*n[1]+1, div(nsample, 2))
rand_y_start_list = rand(1:size(v6, 2)-factor*n[1]+1, div(nsample, 2))

# set up jutul model

function VtoK_nowater(v::Matrix{T}, d::Tuple{T, T}; α::T=T(20)) where T

    n = size(v)
    idx_ucfmt = find_water_bottom((v.-T(3.5)).*(v.>T(3.5)))
    capgrid = Int(round(T(50)/d[2]))

    return vcat([vcat(
        α * exp.(v[i,1:idx_ucfmt[i]-capgrid-1]) .- α*exp(T(1.48)),
        T(1e-3) * ones(T, capgrid),
        α*exp.(v[i,idx_ucfmt[i]:end])  .- α*exp(T(3.5)))' for i = 1:n[1]]...)
end

ϕ = 0.25
K = max.(Float64.(VtoK_nowater(1.0./downsample(1.0./v6[1:256, 1, end-160+1:end], factor), (d[1], d[end])) * md), 1e-3 * md);
model = jutulModel(n, d, ϕ, K1to3(K; kvoverkh=0.36); h=h)

## simulation time steppings
tstep = 365.25 * ones(18)
tot_time = sum(tstep)
nt = length(tstep)

## injection & production
inj_loc = (div(n[1], 2), 1, n[end]-20) .* d
irate = 0.3
f = jutulForce(irate, [inj_loc])

## set up modeling operator
S = jutulModeling(model, tstep)

## simulation
mesh = CartesianMesh(model)
T(x) = log.(KtoTrans(mesh, K1to3(exp.(x); kvoverkh=0.36)))

logK = log.(K)

@time state = S(T(logK), f)

logKs = zeros(n[1], n[end], nsample);
for i = 1:div(nsample, 2)
    println(i)
    K = max.(Float64.(VtoK_nowater(1.0./downsample(1.0./v6[rand_x_start_list[i]:rand_x_start_list[i]+255, rand_y_start_list[i], end-160+1:end], factor), (d[1], d[end])) * md), 1e-3 * md);
    logKs[:,:,2*i-1] = log.(K)
    K = max.(Float64.(VtoK_nowater(1.0./downsample(1.0./v6[rand_x_start_list[i], rand_y_start_list[i]:rand_y_start_list[i]+255, end-160+1:end], factor), (d[1], d[end])) * md), 1e-3 * md);
    logKs[:,:,2*i] = log.(K)
end

conc = zeros(n[1], n[end], nt, nsample);
pres = zeros(n[1], n[end], nt, nsample);

for i = 1:nsample

    Base.flush(Base.stdout)

    println("sample $(i)")
    @time state = S(T(logKs[:,:,i]), f)
    conc[:,:,:,i] = reshape(Saturations(state), n[1], n[end], nt)
    pres[:,:,:,i] = reshape(Pressure(state), n[1], n[end], nt)
end

save_dict = @strdict n d nsample h ϕ tstep inj_loc irate logKs conc pres
@tagsave(
    joinpath(datadir(sim_name, exp_name), savename(save_dict, "jld2"; digits=6)),
    save_dict;
    safe=true
)

