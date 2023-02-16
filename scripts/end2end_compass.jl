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
using Random
Random.seed!(2023)

include(srcdir("dummy_src_file.jl"))

sim_name = "end2end-inv"
exp_name = "compass"

mkpath(datadir())
mkpath(plotsdir())

## grid size
JLD2.@load datadir("BGCompass_tti_625m.jld2") m d rho;
vp = 1f0./sqrt.(m)
d = (6., 6.)
n = size(m)

# downsample
cut_area = [201, 450, 182, n[end]]
m = m[cut_area[1]:cut_area[2],cut_area[3]:cut_area[4]]
h = (cut_area[3]-1) * d[end]
v = Float64.(sqrt.(1f0./m));
factor = 2
v = 1.0./downsample(1.0./v, factor)
Kh = VtoK(v, d);
K = Float64.(Kh * md);

# dimension for fluid-flow
ns = (size(K,1), 1, size(K,2))
ds = (d[1] * factor, 2000.0, d[2] * factor)

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
S = jutulModeling(model, tstep)

## simulation
mesh = CartesianMesh(model)
T(x) = log.(KtoTrans(mesh, K1to3(exp.(x); kvoverkh=0.36)))

logK = log.(K)

@time state = S(T(logK), f)

### observed states
nv = length(tstep)
function O(state::AbstractVector)
    full_his = Float32.(reshape(state[1:nv*prod(ns)], ns[1], ns[end], nv))
    return [full_his[:,:,i] for i = 1:nv]
end
sw_true = O(state)

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

timeS = timeR = 2000f0               # recording time
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
f0 = 0.05f0     # kHz
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
misc_dict = @strdict nsrc nrec nv cut_area tstep factor d n

### generate/load data
if ~isfile(datadir("seismic-data", savename(misc_dict, "jld2"; digits=6)))
    println("generating data")
    global d_obs = [Fs[i]*q for i = 1:nv]
    seismic_dict = @strdict nsrc nrec nv cut_area tstep factor d n d_obs q srcGeometry recGeometry model
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

lower, upper = 1.1*minimum(logK), 0.9*maximum(logK)
box_logK(x::AbstractArray{T}) where T = max.(min.(x,T(upper)),T(lower))

# Main loop
niterations = 100
fhistory = zeros(niterations)

### add box to co2 and velocity
box_co2(x::AbstractArray{T}) where T = max.(min.(x,T(1)),T(0))
box_co2(x::AbstractVector) = [box_co2(x[i]) for i = 1:length(x)]
box_v(x::AbstractMatrix{T}) where T = max.(min.(x,T(maximum(vp))),T(minimum(vp)))
box_v(x::AbstractVector) = [box_v(x[i]) for i = 1:length(x)]

### inversion initialization
logK0 = deepcopy(logK)
logK0[v.>3.5] .= mean(logK[v.>3.5])
logK_init = deepcopy(logK0)
y_init = box_co2(O(S(T(logK_init), f)))

# objective function for inversion
function obj(logK)
    c = box_co2(O(S(T(logK), f))); v = R(pad(c)); v_up = box_v(v); dpred = F(v_up);
    fval = .5f0 * norm(dpred-d_obs)^2f0/nsrc/nv
    @show fval
    return fval
end

# ADAM-W algorithm
learning_rate = 1e-2
nssample = 4
opt = Flux.Optimise.ADAMW(learning_rate, (0.9, 0.999), 1e-4)

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
    function obj(logK)
        c = box_co2(O(S(T(box_logK(logK)), f))); v = R(pad(c)); v_up = box_v(v); dpred = F(v_up);
        fval = .5f0 * norm(dpred-dobs)^2f0/nsrc/nv
        @show fval
        return fval
    end
    ## AD by Flux
    @time fval, gs = Flux.withgradient(() -> obj(logK0), Flux.params(logK0))

    fhistory[j] = fval
    g = gs[logK0]
    for p in Flux.params(logK0)
        Flux.Optimise.update!(opt, p, gs[p])
    end
        
    println("Inversion iteration no: ",j,"; function value: ", fhistory[j])

    ### plotting
    y_predict = box_co2(O(S(T(logK0), f)))

    ### save intermediate results
    save_dict = @strdict j nssample logK0 g step niterations nv nsrc nrec nv cut_area tstep factor n d fhistory
    @tagsave(
        joinpath(datadir(sim_name, exp_name), savename(save_dict, "jld2"; digits=6)),
        save_dict;
        safe=true
    )

    ## save figure
    fig_name = @strdict j nssample logK0 step niterations nv nsrc nrec nv cut_area tstep factor n d fhistory

    ## compute true and plot
    SNR = -2f1 * log10(norm(K-exp.(logK0))/norm(K))
    fig = figure(figsize=(20,12));
    subplot(2,2,1);
    imshow(exp.(logK0)'./md,vmin=0,vmax=maximum(K/md));title("inversion by NN, $(j) iter");colorbar();
    subplot(2,2,2);
    imshow(K'./md,vmin=0,vmax=maximum(K/md));title("GT permeability");colorbar();
    subplot(2,2,3);
    imshow(exp.(logK_init)'./md,vmin=0,vmax=maximum(K/md));title("initial permeability");colorbar();
    subplot(2,2,4);
    imshow(abs.(K'-exp.(logK0)')./md,vmin=0,vmax=maximum(K/md));title("abs error, SNR=$SNR");colorbar();
    suptitle("End-to-end Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_K.png"), fig);
    close(fig)

    ## loss
    fig = figure(figsize=(20,12));
    plot(fhistory[1:j]);title("loss=$(fhistory[j])");
    suptitle("End-to-end Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_loss.png"), fig);
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
        imshow(y_predict[3*i]', vmin=0, vmax=1);
        title("predict at snapshot $(3*i)")
        subplot(4,5,i+15);
        imshow(5*abs.(sw_true[3*i]'-y_predict[3*i]'), vmin=0, vmax=1);
        title("5X diff at snapshot $(3*i)")
    end
    suptitle("End-to-end Inversion at iter $j")
    tight_layout()
    safesave(joinpath(plotsdir(sim_name, exp_name), savename(fig_name; digits=6)*"_saturation.png"), fig);
    close(fig)

end
