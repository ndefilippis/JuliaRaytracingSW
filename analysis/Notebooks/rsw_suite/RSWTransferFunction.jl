using JLD2
using FourierFlows
using FourierFlows: parsevalsum2
using Printf
using CairoMakie
using AbstractFFTs
include("RSWUtils.jl")
include("AnalysisUtils.jl")

function compute_transfer_function(run_directory, grid)
    Nsnapshots = count_key_snapshots(run_directory, "rsw")
    
    f0, Cg2 = read_rsw_params(run_directory)
    params = (; f=f0, Cg2)
    ν, nν = read_rsw_dissipation(run_directory)
    
   
    Eh_flux = zeros(grid.nkr, grid.nl)
    Egggh_flux = zeros(grid.nkr, grid.nl)
    Eggwh_flux = zeros(grid.nkr, grid.nl)
    Egwwh_flux = zeros(grid.nkr, grid.nl)
    Ewwwh_flux = zeros(grid.nkr, grid.nl)

    Zh_flux = zeros(grid.nkr, grid.nl)
    Zgggh_flux = zeros(grid.nkr, grid.nl)
    Zggwh_flux = zeros(grid.nkr, grid.nl)
    Zgwwh_flux = zeros(grid.nkr, grid.nl)
    Zwwwh_flux = zeros(grid.nkr, grid.nl)

    snap_frames = 2:Nsnapshots
    
    times = zeros(length(snap_frames))
    array_idx = 1
    
    Φ₀, Φ₊, Φ₋ = compute_balanced_wave_bases(grid, (; f=f0, Cg2))
    
    for snap_idx = snap_frames
        if (snap_idx % 100 == 0)
            println(@sprintf("Computed frame %d/%d", snap_idx, Nsnapshots))
        end
        
        t, rsw_sol = load_key_snapshot(run_directory, "rsw", snap_idx)
        times[array_idx] = t
        
        dealias!(rsw_sol, grid)
        
        uh = rsw_sol[:,:,1]
        vh = rsw_sol[:,:,2]
        ηh = rsw_sol[:,:,3]

        ((ugh, vgh, ηgh), (uwh, vwh, ηwh)) = wave_balanced_decomposition(uh, vh, ηh, grid, params)
        
        total_field = compute_derivatives(uh,  vh,  ηh,  grid, params)
        geo_field   = compute_derivatives(ugh, vgh, ηgh, grid, params)
        wave_field  = compute_derivatives(uwh, vwh, ηwh, grid, params)

        qh  = total_field[end]
        qgh = geo_field[end]
        qwh = wave_field[end]

        total_quadratic_terms = compute_duvηdth(total_field, total_field, grid)
        gg_quadratic_terms = compute_duvηdth(geo_field,  geo_field,  grid)
        gw_quadratic_terms = compute_duvηdth(geo_field,  wave_field, grid) .+ compute_duvηdth(wave_field, geo_field, grid)
        ww_quadratic_terms = compute_duvηdth(wave_field, wave_field, grid)

        E, Z = compute_flux_fields(uh, vh, ηh, qh, total_quadratic_terms, grid, params)
        Eggg, Zggg = compute_flux_fields(ugh, vgh, ηgh, qgh, gg_quadratic_terms, grid, params)
        
        Eggw, Zggw = compute_flux_fields(ugh, vgh, ηgh, qgh, gw_quadratic_terms, grid, params) 
        Ewgg, Zwgg = compute_flux_fields(uwh, vwh, ηwh, qwh, gg_quadratic_terms, grid, params)
        
        Egww, Zgww = compute_flux_fields(ugh, vgh, ηgh, qgh, ww_quadratic_terms, grid, params)
        Ewwg, Zwwg = compute_flux_fields(uwh, vwh, ηwh, qwh, gw_quadratic_terms, grid, params)
        
        Ewww, Zwww = compute_flux_fields(uwh, vwh, ηwh, qwh, ww_quadratic_terms, grid, params)

        @. Eh_flux    += E
        @. Egggh_flux += Eggg
        @. Eggwh_flux += Eggw + Ewgg
        @. Egwwh_flux += Egww + Ewwg
        @. Ewwwh_flux += Ewww

        @. Zh_flux    += Z
        @. Zgggh_flux += Zggg
        @. Zggwh_flux += Zggw + Zwgg
        @. Zgwwh_flux += Zgww + Zwwg
        @. Zwwwh_flux += Zwww

        array_idx += 1
    end
    return times, Eh_flux, Egggh_flux, Eggwh_flux, Egwwh_flux, Ewwwh_flux, Zh_flux, Zgggh_flux, Zggwh_flux, Zgwwh_flux, Zwwwh_flux
end

function compute_derivatives(uh, vh, ηh, grid, params)
    uxh = @. 1im * grid.kr * uh
    vxh = @. 1im * grid.kr * vh
    ηxh = @. 1im * grid.kr * ηh
    uyh = @. 1im * grid.l  * uh
    vyh = @. 1im * grid.l  * vh
    ηyh = @. 1im * grid.l  * ηh

    qh = @. 1im * grid.kr * vh - 1im * grid.l * uh - params.f * ηh
    
    u  = irfft(uh, grid.nx)
    v  = irfft(vh, grid.nx)
    η  = irfft(ηh, grid.nx)
    ux = irfft(uxh, grid.nx)
    vx = irfft(vxh, grid.nx)
    ηx = irfft(ηxh, grid.nx)
    uy = irfft(uyh, grid.nx)
    vy = irfft(vyh, grid.nx)
    ηy = irfft(ηyh, grid.nx)

    return (u, v, η, ux, vx, ηx, uy, vy, ηy, qh)
end
function compute_duvηdth(field_set_1, field_set_2, grid)
    u1, v1, η1, ux1, vx1, ηx1, uy1, vy1, ηy1 = field_set_1
    u2, v2, η2, ux2, vx2, ηx2, uy2, vy2, ηy2 = field_set_2
    uuxh = rfft(u1 .* ux2)
    vuyh = rfft(v1 .* uy2)
    uvxh = rfft(u1 .* vx2)
    vvyh = rfft(v1 .* vy2)
    uηxh = rfft(u1 .* ηx2)
    vηyh = rfft(v1 .* ηy2)
    ηuh  = rfft(η1 .* u2)
    ηvh  = rfft(η1 .* v2)
    divηuh   = @. 1im * grid.kr * ηuh + 1im * grid.l * ηvh

    dudth = @. (-uuxh - vuyh)# + D * uh + f0 * vh - Cg2 * ηxh)
    dvdth = @. (-uvxh - vvyh)# + D * vh - f0 * uh - Cg2 * ηyh)
    dηdth = @. (-divηuh)# + D * ηh - uxh - vyh)

    return dudth, dvdth, dηdth
    
end

function compute_flux_fields(uh, vh, ηh, qh, quadratic_terms, grid, params)
    dudth, dvdth, dηdth = quadratic_terms
    Euh = @. conj(uh) * dudth
    Evh = @. conj(vh) * dvdth
    Eηh = @. conj(ηh) * dηdth
    
    Eh = @. real(0.5*(Euh + Evh) + 0.5 * params.Cg2 * Eηh)
    Zh = @. real(conj(qh) * (1im * grid.kr * dvdth - 1im * grid.l * dudth - params.f * dηdth))

    return Eh, Zh
end