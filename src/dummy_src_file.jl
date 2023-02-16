export VtoK, downsample, Patchy, jitter

function VtoK(v::Matrix{T}, d::Tuple{T, T}; α::T=T(20)) where T

    n = size(v)
    idx_wb = find_water_bottom(v.-minimum(v))
    idx_ucfmt = find_water_bottom((v.-T(3.5)).*(v.>T(3.5)))
    Kh = zeros(T, n)
    capgrid = Int(round(T(50)/d[2]))
    for i = 1:n[1]
        Kh[i,1:idx_wb[i]-1] .= T(1e-10)  # water layer
        Kh[i,idx_wb[i]:idx_ucfmt[i]-capgrid-1] .= α*exp.(v[i,idx_wb[i]:idx_ucfmt[i]-capgrid-1])
        Kh[i,idx_ucfmt[i]-capgrid:idx_ucfmt[i]-1] .= T(1e-3)
        Kh[i,idx_ucfmt[i]:end] .= α*exp.(v[i,idx_ucfmt[i]:end]) .- α*exp(T(3.5))
    end
    return Kh
end

function downsample(v::Matrix{T}, factor::Int) where T
    v_out_size = div.(size(v), factor)
    v_out = zeros(T, v_out_size)
    for i = 1:v_out_size[1]
        for j = 1:v_out_size[2]
            v_out[i,j] = mean(v[factor*i-factor+1:factor*i, factor*j-factor+1:factor*j])
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
