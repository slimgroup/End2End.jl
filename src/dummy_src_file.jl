export VtoK, downsample, Patchy, jitter, box_logK, box_co2, box_v, dummy_JUDI_operation

function VtoK(v::Matrix{T}, d::Tuple{T, T}; α::T=T(20)) where T

    n = size(v)
    idx_wb = find_water_bottom(v.-minimum(v))
    idx_ucfmt = find_water_bottom((v.-T(3.5)).*(v.>T(3.5)))
    capgrid = Int(round(T(50)/d[2]))

    return vcat([vcat(
        T(1e-10) * ones(Float32, idx_wb[i]-1),
        α * exp.(v[i,idx_wb[i]:idx_ucfmt[i]-capgrid-1]) .- α*exp(T(1.48)),
        T(1e-3) * ones(T, capgrid),
        α*exp.(v[i,idx_ucfmt[i]:end])  .- α*exp(T(3.5)))' for i = 1:n[1]]...)
end

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

function downsample(v::Matrix{T}, factor::Int) where T
    return downsample(v, (factor, factor))
end

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

#### Patchy saturation model

function Patchy(sw::Matrix{T}, vp::Matrix{T}, rho::Matrix{T}, phi::Matrix{T};
    bulk_min::T=T(36.6f9), bulk_fl1::T=T(2.735f9), bulk_fl2::T=T(0.125f9), ρw::T=T(700f0), ρo::T=T(1000.0f0)) where T

    ### works for channel problem
    vs = vp./sqrt(3f0)
    bulk_sat1 = rho .* (vp.^2f0 - 4f0/3f0 .* vs.^2f0)
    shear_sat1 = rho .* (vs.^2f0)

    patch_temp = bulk_sat1 ./(bulk_min .- bulk_sat1) .- 
    bulk_fl1 ./ phi ./ (bulk_min .- bulk_fl1) .+ 
    bulk_fl2 ./ phi ./ (bulk_min .- bulk_fl2)

    bulk_sat2 = bulk_min./(1f0./patch_temp .+ 1f0)

    bulk_new = 1f0./( (1f0.-sw)./(bulk_sat1+4f0/3f0*shear_sat1) 
    + sw./(bulk_sat2+4f0/3f0*shear_sat1) ) - 4f0/3f0*shear_sat1

    rho_new = rho + phi .* sw * (ρw - ρo)

    Vp_new = sqrt.((bulk_new+4f0/3f0*shear_sat1)./rho_new)
    return Vp_new, rho_new

end

function Patchy(sw::Array{T, 3}, vp::Matrix{T}, rho::Matrix{T}, phi::Matrix{T};
    bulk_min::T=T(36.6f9), bulk_fl1::T=T(2.735f9), bulk_fl2::T=T(0.125f9), ρw::T=T(700f0), ρo::T=T(1000.0f0)) where T

    stack = [Patchy(sw[i,:,:], vp, rho, phi; bulk_min=bulk_min, bulk_fl1=bulk_fl1, bulk_fl2=bulk_fl2, ρw = ρw, ρo=ρo) for i = 1:size(sw,1)]
    return [stack[i][1] for i = 1:size(sw,1)], [stack[i][2] for i = 1:size(sw,1)]
end

function Patchy(sw::Vector{Matrix{T}}, vp::Matrix{T}, rho::Matrix{T}, phi::Matrix{T};
    bulk_min::T=T(36.6f9), bulk_fl1::T=T(2.735f9), bulk_fl2::T=T(0.125f9), ρw::T=T(700f0), ρo::T=T(1000.0f0)) where T

    stack = [Patchy(sw[i], vp, rho, phi; bulk_min=bulk_min, bulk_fl1=bulk_fl1, bulk_fl2=bulk_fl2, ρw = ρw, ρo=ρo) for i = 1:size(sw,1)]
    return [stack[i][1] for i = 1:size(sw,1)], [stack[i][2] for i = 1:size(sw,1)]
end

function jitter(nsrc::Int, nssample::Int)
    npatch = Int(nsrc/nssample)
    return rand(1:npatch, nssample) .+ convert(Vector{Int},0:npatch:(nsrc-1))
end

box_logK(x::AbstractArray{T}; upper=log(6000*md), lower=log(1e-3*md)) where T = max.(min.(x,T(upper)),T(lower))
box_co2(x::AbstractArray{T}) where T = max.(min.(x,T(0.9)),T(0))
box_co2(x::AbstractVector) = [box_co2(x[i]) for i = 1:length(x)]
box_v(x::AbstractMatrix{T}; upper=4.7f0, lower=1.48f0) where T = max.(min.(x,T(upper)),T(lower))
box_v(x::AbstractVector) = [box_v(x[i]) for i = 1:length(x)]

function dummy_JUDI_operation()
## grid size
n = (20, 20)
m = 1f0./1.5f0^2 * ones(Float32, n)
d = (6., 6.)
o = (0f0, 0f0)          # origin

nsrc = 1       # num of sources
nrec = 2      # num of receivers

model = Model(n, d, o, m)

timeS = timeR = 800f0               # recording time
dtS = dtR = 4f0                     # recording time sampling rate

xsrc = convertToCell(1f0)
zsrc = convertToCell(0f0)
xrec = range(1f0,2f0,length=nrec)
zrec = range(1f0,2f0,length=nrec)
ysrc = convertToCell(0f0)
yrec = 0f0

# set up src/rec geometry
srcGeometry = Geometry(xsrc, ysrc, zsrc; dt=dtS, t=timeS)
recGeometry = Geometry(xrec, yrec, zrec; dt=dtR, t=timeR, nsrc=nsrc)

# set up source
f0 = 0.02f0     # kHz
wavelet = ricker_wavelet(timeS, dtS, f0)
q = judiVector(srcGeometry, wavelet)

# set up simulation operators
F = judiModeling(model, srcGeometry, recGeometry)
dobs = F * q
return dobs
end
