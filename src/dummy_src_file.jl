export VtoK, downsample, Patchy, jitter, box_logK, box_co2, box_v

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

function downsample(v::Matrix{T}, factor::Int) where T
    v_out_size = div.(size(v), factor)
    return vcat([vcat([mean(v[factor*i-factor+1:factor*i, factor*j-factor+1:factor*j]) for j = 1:v_out_size[2]]...)' for i = 1:v_out_size[1]]...)
end

#### Patchy saturation model

function Patchy(sw::Matrix{T}, vp::Matrix{T}, rho::Matrix{T}, phi::Matrix{T};
    bulk_min::T=T(36.6f9), bulk_fl1::T=T(2.735f9), bulk_fl2::T=T(0.125f9), ρw::T=T(700f0), ρo::T=T(1000.0f0)) where T

    ### works for channel problem
    vs = vp./sqrt(3f0)
    bulk_sat1 = rho .* (vp.^2f0 - 4f0/3f0 .* vs.^2f0)
    shear_sat1 = rho .* (vs.^2f0)

    patch_temp = bulk_sat1 ./(bulk_min .- bulk_sat1) - 
    bulk_fl1 ./ phi ./ (bulk_min .- bulk_fl1) + 
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

box_logK(x::AbstractArray{T}; upper=log(1500*md), lower=log(1e-4*md)) where T = max.(min.(x,T(upper)),T(lower))
box_co2(x::AbstractArray{T}) where T = max.(min.(x,T(0.9)),T(0))
box_co2(x::AbstractVector) = [box_co2(x[i]) for i = 1:length(x)]
box_v(x::AbstractMatrix{T}; upper=4.6454024f0, lower=1.48f0) where T = max.(min.(x,T(upper)),T(lower))
box_v(x::AbstractVector) = [box_v(x[i]) for i = 1:length(x)]
