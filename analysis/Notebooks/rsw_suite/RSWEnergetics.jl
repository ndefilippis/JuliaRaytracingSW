using JLD2
using FourierFlows
using FourierFlows: parsevalsum2
using Printf
using CairoMakie
using AbstractFFTs
include("RSWUtils.jl")
include("AnalysisUtils.jl")

function compute_energy_data(run_directory, grid)
    Nsnapshots = count_key_snapshots(run_directory, "rsw")
    
    f, Cg2 = read_rsw_params(run_directory)
    params = (; f, Cg2)

    Eh  = zeros(grid.nkr, grid.nl)
    Egh = zeros(grid.nkr, grid.nl)
    Ewh = zeros(grid.nkr, grid.nl)
    KEh   = zeros(grid.nkr, grid.nl)
    KEgh  = zeros(grid.nkr, grid.nl)
    KEwh  = zeros(grid.nkr, grid.nl)
    APEh  = zeros(grid.nkr, grid.nl)
    APEgh = zeros(grid.nkr, grid.nl)
    APEwh = zeros(grid.nkr, grid.nl)
    Zh  = zeros(grid.nkr, grid.nl)

    snap_frames = 2:Nsnapshots
    N_frames = length(snap_frames)
    
    times = zeros(N_frames)

    KE_total   = zeros(N_frames)
    KEg_total  = zeros(N_frames)
    KEw_total  = zeros(N_frames)
    APE_total  = zeros(N_frames)
    APEg_total = zeros(N_frames)
    APEw_total = zeros(N_frames)
    Z_total   = zeros(N_frames)
    
    array_idx = 1
    
    bases = compute_balanced_wave_bases(grid, params)
    for snap_idx=snap_frames
        if (snap_idx % 100 == 0)
            println(@sprintf("Computed frame %d/%d", snap_idx, Nsnapshots))
        end
        
        t, rsw_sol = load_key_snapshot(run_directory, "rsw", snap_idx)
        times[array_idx] = t
        
        dealias!(rsw_sol, grid)
        ((KE, PE, KEg, PEg, KEw, PEw, Eg, Ew, Z), (KE_sum, PE_sum, KEg_sum, PEg_sum, KEw_sum, PEw_sum, Z_sum)) = compute_energy(rsw_sol, bases, grid, params)

        @. Egh   += Eg
        @. Ewh   += Ew
        @. KEh   += KE
        @. KEgh  += KEg
        @. KEwh  += KEw
        @. APEh  += PE
        @. APEgh += PEg
        @. APEwh += PEw
        @. Zh    += Z

        KE_total[array_idx]   = KE_sum
        KEg_total[array_idx]  = KEg_sum
        KEw_total[array_idx]  = KEw_sum
        APE_total[array_idx]  = PE_sum
        APEg_total[array_idx] = PEg_sum
        APEw_total[array_idx] = PEw_sum
        Z_total[array_idx]    = Z_sum
        
        array_idx += 1
    end
    
    return times, Egh, Ewh, KEh, KEgh, KEwh, APEh, APEgh, APEwh, Zh, KE_total, KEg_total, KEw_total, APE_total, APEg_total, APEw_total, Z_total  
end

function compute_energy(snapshot, bases, grid, params)
    uh = snapshot[:,:,1]
    vh = snapshot[:,:,2]
    ηh = snapshot[:,:,3]
    
    c₀, c₊, c₋ = compute_balanced_wave_weights(uh, vh, ηh, bases..., params)
    ((ugh, vgh, ηgh), (uwh, vwh, ηwh)) = wave_balanced_decomposition(uh, vh, ηh, grid, params)
    qh = @. 1im * grid.kr * vh - 1im * grid.l * uh - params.f * ηh

    ((KE,  KE_total),  (PE,  PE_total))  = compute_energetics(uh, vh, ηh, grid, params)
    ((KEg, KEg_total), (PEg, PEg_total)) = compute_energetics(ugh, vgh, ηgh, grid, params)
    Z, Z_total                           = compute_enstrophy(qh, grid, params)
    ((KEw, KEw_total), (PEw, PEw_total)) = compute_energetics(uwh, vwh, ηwh, grid, params)

    Eg = abs2.(c₀) * grid.Lx * grid.Ly / (grid.nx^2 * grid.ny^2)
    Ew = (abs2.(c₊) + abs2.(c₋)) * grid.Lx * grid.Ly / (grid.nx^2 * grid.ny^2)

    return ((KE, PE, KEg, PEg, KEw, PEw, Eg, Ew, Z), (KE_total, PE_total, KEg_total, PEg_total, KEw_total, PEw_total, Z_total))
end

function compute_energetics(uh, vh, ηh, grid, params)
    KE = 0.5 * (abs2.(uh) + abs2.(vh)) * grid.Lx * grid.Ly / (grid.nx^2 * grid.ny^2)
    PE = 0.5 * params.Cg2 * abs2.(ηh)  * grid.Lx * grid.Ly / (grid.nx^2 * grid.ny^2)

    KE_total = 0.5 * (parsevalsum2(uh, grid) + parsevalsum2(vh, grid)) / grid.Lx / grid.Ly
    PE_total = 0.5 * params.Cg2 * parsevalsum2(ηh, grid) / grid.Lx / grid.Ly
    return ((KE, KE_total), (PE, PE_total))
end

function compute_enstrophy(qh, grid, params)
    Z = abs2.(qh)  * grid.Lx * grid.Ly / (grid.nx^2 * grid.ny^2)
    Z_total = parsevalsum2(Z, grid) / grid.Lx / grid.Ly

    return Z, Z_total
end