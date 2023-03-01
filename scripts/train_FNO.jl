# author: Ziyi Yin, ziyi.yin@gatech.edu
# This script trains a Fourier Neural Operator which maps 2D permeability distribution to time-varying CO2 concentration snapshots.
# The PDE is in 2D while FNO requires 3D FFT

using DrWatson
@quickactivate "jutul-compass"

using Pkg; Pkg.instantiate();
using FNO4CO2
using PyPlot
using JLD2
using Flux, Random, FFTW
using Statistics, LinearAlgebra
using InvertibleNetworks:ActNorm
matplotlib.use("Agg")

if gpu_flag
    device = gpu
else
    device = cpu
end

Random.seed!(1234)

# Define raw data directory
data_path = "/slimdata/zyin62/jutul-compass/h=1086.0_irate=0.3_nsample=6000_ϕ=0.25.jld2"
mkpath("/slimdata/zyin62/jutul-compass")
# Download the dataset into the data directory if it does not exist
if ~isfile(data_path)
    run(`wget https://www.dropbox.com/s/fnvpodx15j8hbrs/'
        'h=1086.0_irate=0.3_nsample=6000_ϕ=0.25.jld2 -q -O $data_path`)
end

JLD2.@load data_path n d nsample tstep logKs conc
logKs = Float32.(logKs)
conc = Float32.(conc)

nsamples = size(logKs, 3)

ntrain = 5960
nvalid = 40

batch_size = 8
learning_rate = 1f-4

epochs = 1000

modes = [16, 16, 4]
width = 20

n = (n[1], n[end])
d = 1f0./n

s = 1

nt = size(conc,3)
dt = 1f0/(nt-1)

AN = ActNorm(ntrain)
AN.forward(reshape(logKs[1:s:end,1:s:end,1:ntrain], n[1], n[2], 1, ntrain));

y_train = conc[1:s:end,1:s:end,:,1:ntrain];
y_valid = conc[1:s:end,1:s:end,:,ntrain+1:ntrain+nvalid];

grid = gen_grid(n, d, nt, dt)

x_train = perm_to_tensor(logKs[1:s:end,1:s:end,1:ntrain],grid,AN);
x_valid = perm_to_tensor(logKs[1:s:end,1:s:end,ntrain+1:ntrain+nvalid],grid,AN);

# value, x, y, t

NN = Net3d(modes, width) |> device;

Flux.trainmode!(NN, true);
w = Flux.params(NN);

opt = Flux.Optimise.ADAMW(learning_rate, (0.9f0, 0.999f0), 1f-4);
nbatches = Int(ntrain/batch_size)

Loss = zeros(Float32,epochs*nbatches)
Loss_valid = zeros(Float32, epochs)

# plot figure
x_plot = x_valid[:, :, :, :, 1:1]
y_plot = y_valid[:, :, :, 1:1]

# Define result directory

sim_name = "3D_FNO"
exp_name = "2phaseflow-jutul-compass"
plot_path = plotsdir(sim_name, exp_name)
save_path = joinpath("/slimdata/zyin62/jutul-compass/data/", sim_name, exp_name)
mkpath(save_path)

## training
save_network_every = 10

for ep = 1:epochs

    Base.flush(Base.stdout)
    idx_e = reshape(randperm(ntrain), batch_size, nbatches)

    Flux.trainmode!(NN, true);
    for b = 1:nbatches
        x = x_train[:, :, :, :, idx_e[:,b]] |> device;
        y = y_train[:, :, :, idx_e[:,b]] |> device;
        grads = gradient(w) do
            global loss = norm(clamp.(NN(x), 0f0, 1f0)-y)/norm(y)
            return loss
        end
        Loss[(ep-1)*nbatches+b] = loss
        for p in w
            Flux.Optimise.update!(opt, p, grads[p])
        end
    end

    Flux.testmode!(NN, true);

    y_predict = clamp.(NN(x_plot |> device), 0f0, 1f0)   |> cpu

    fig = figure(figsize=(20, 12))

    for i = 1:5
        subplot(4,5,i)
        imshow(x_plot[:,:,3*i,1,1]')
        title("x")

        subplot(4,5,i+5)
        imshow(y_plot[:,:,3*i,1]', vmin=0, vmax=1)
        title("true y")

        subplot(4,5,i+10)
        imshow(y_predict[:,:,3*i,1]', vmin=0, vmax=1)
        title("predict y")

        subplot(4,5,i+15)
        imshow(5f0 .* abs.(y_plot[:,:,3*i,1]'-y_predict[:,:,3*i,1]'), vmin=0, vmax=1)
        title("5X abs difference")

    end
    tight_layout()
    fig_name = @strdict ep batch_size Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_fitting.png"), fig);
    close(fig)

    NN_save = NN |> cpu;
    w_save = Flux.params(NN_save)   

    Loss_valid[ep] = norm(clamp.(NN_save(x_valid), 0f0, 1f0) - y_valid)/norm(y_valid)

    loss_train = Loss[1:ep*nbatches]
    loss_valid = Loss_valid[1:ep]
    fig = figure(figsize=(20, 12))
    subplot(1,3,1)
    plot(loss_train)
    title("training loss at epoch $ep")
    subplot(1,3,2)
    plot(1:nbatches:nbatches*ep, loss_valid); 
    title("validation loss at epoch $ep")
    subplot(1,3,3)
    plot(loss_train);
    plot(1:nbatches:nbatches*ep, loss_valid); 
    xlabel("iterations")
    ylabel("value")
    title("Objective function at epoch $ep")
    legend(["training", "validation"])
    tight_layout();
    safesave(joinpath(plot_path, savename(fig_name; digits=6)*"_3Dfno_loss.png"), fig);
    close(fig); 

    if mod(ep, save_network_every) == 0
    param_dict = @strdict ep NN_save w_save batch_size Loss modes width learning_rate epochs s n d nt dt AN ntrain nvalid loss_train loss_valid
    @tagsave(
        joinpath(save_path, savename(param_dict, "jld2"; digits=6)),
        param_dict;
        safe=true
    )
    end
    
end