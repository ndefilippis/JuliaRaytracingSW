using .RotatingShallowWater
using .SequencedOutputs
using FourierFlows
using Printf
using LinearAlgebra: mul!, ldiv!
using JLD2

import .Parameters

function load_initial_condition_from_file!(prob, snapshot_filename, snapshot_key)
    snapshot_file = jldopen(snapshot_filename, "r")
    load_from_snapshot!(prob, snapshot_file[snapshot_key])
    JLD2.close(snapshot_file)
end

function load_from_snapshot!(prob, initial_snapshot)
    grid = prob.grid
    dev = typeof(grid.device)
    T = typeof(grid.Lx)

    A = typeof(prob.vars.uh)
    
    device_snapshot = device_array(grid.device)(initial_snapshot)
    
    snapshot_nkr = size(device_snapshot, 1)
    snapshot_nl  = size(device_snapshot, 2)
    half_nl = snapshot_nkr - 1
    scale_factor = grid.nl^2 / snapshot_nl^2
    
    @devzeros dev Complex{T} (grid.nkr, grid.nl, 3) new_snapshot
    
    @views new_snapshot[1:snapshot_nkr, 1:half_nl, :]           .= scale_factor * device_snapshot[:, 1:half_nl, :]
    @views new_snapshot[1:snapshot_nkr, (end-half_nl+1):end, :] .= scale_factor * device_snapshot[:, (half_nl+1):end, :]

    RotatingShallowWater.set_solution!(prob, new_snapshot[:,:,1], new_snapshot[:,:,2], new_snapshot[:,:,3])
end

function set_shafer_initial_condition!(prob, Kg, Kw, ag, aw, f, Cg2)
    grid = prob.grid
    dev = typeof(grid.device)
    T = typeof(grid.Lx)

    @devzeros dev Complex{T} (grid.nkr, grid.nl) ugh vgh ηgh uwh vwh ηwh
    @devzeros dev T (grid.nx, grid.ny) ug uw vg vw ζ
    
    geo_filter  = (Kg[1]^2 .<= grid.Krsq .<= Kg[2]^2) .& (grid.Krsq .> 0)
    wave_filter = (Kw[1]^2 .<= grid.Krsq .<= Kw[2]^2) .& (grid.Krsq .> 0)
    phase = device_array(grid.device)(2π*rand(grid.nkr, grid.nl))
    sgn =  device_array(grid.device)(sign.(rand(grid.nkr, grid.nl) .- 0.5))
    shift = exp.(1im * phase)
    ω = sqrt.(f^2 .+ Cg2 * grid.Krsq)
    geo_amp_factor  = 1 ./ ω# .* (grid.invKrsq).^(3/4) # Uncomment for constant radial vorticity scaling
    wave_amp_factor = sqrt.(grid.invKrsq) ./ (2 * ω)# .* (grid.invKrsq).^(1/4) # Uncomment for constant radial energy scaling
    ηgh[geo_filter] += ( geo_amp_factor  * f .* shift)[geo_filter]
    ugh[geo_filter] += (-geo_amp_factor  * 1im * Cg2 .* grid.l  .* shift)[geo_filter]
    vgh[geo_filter] += ( geo_amp_factor  * 1im * Cg2 .* grid.kr .* shift)[geo_filter]

    ldiv!(ug, grid.rfftplan, deepcopy(ugh))
    ldiv!(vg, grid.rfftplan, deepcopy(vgh))
    ζh = 1im * grid.kr .* vgh - 1im * grid.l .* ugh
    ldiv!(ζ, grid.rfftplan, deepcopy(ζh))

    Umax = maximum(sqrt.(ug.^2 + vg.^2))
    ηgh .*= ag / Umax
    ugh .*= ag / Umax
    vgh .*= ag / Umax

    ηwh[wave_filter] += (wave_amp_factor .* grid.Krsq .* shift)[wave_filter]
    uwh[wave_filter] += (wave_amp_factor .* (sgn .* grid.kr .* ω .* shift + 1im * f * grid.l  .* shift))[wave_filter]
    vwh[wave_filter] += (wave_amp_factor .* (sgn .* grid.l  .* ω .* shift - 1im * f * grid.kr .* shift))[wave_filter]
    
    ldiv!(uw, grid.rfftplan, deepcopy(uwh))
    ldiv!(vw, grid.rfftplan, deepcopy(vwh))
    ζh = 1im * grid.kr .* vwh - 1im * grid.l .* uwh
    ldiv!(ζ, grid.rfftplan, deepcopy(ζh))
    Umax = maximum(sqrt.(uw.^2 + vw.^2))
    
    ηwh .*= aw / Umax
    uwh .*= aw / Umax
    vwh .*= aw / Umax
    RotatingShallowWater.set_solution!(prob, ugh + uwh, vgh + vwh, ηgh + ηwh)
end

function initialize_problem()
    dev = GPU()
    Lx=Parameters.L
    nx=Parameters.nx
    dx=Lx/nx
    kmax = (nx/2 - 1) * Lx / (2π) * (1 - Parameters.aliased_fraction)
    nν=Parameters.nν

    umax = Parameters.ag + Parameters.aw
    νtune = Parameters.νtune
    dt = Parameters.cfltune / umax * dx

    # ϵ = 3.3561e-19 # Fixed energy dissipation rate from baseline run
    # ν = Parameters.νtune * (ϵ/(kmax^(6*nν-2)))^(1/3) # Fixed energy dissipation scaling
    ν = Parameters.νtune * dx / (kmax^(2*nν)) / dt # Fixed enstrophy dissipation scaling

    nsteps = ceil(Int, Parameters.T / dt)
    spinup_step = floor(Int, Parameters.spinup_T / dt)
    output_freq = max(floor(Int, Parameters.output_dt / dt), 1)
    diags_freq = max(floor(Int, Parameters.diag_dt / dt), 1)

    println(@sprintf("Total steps: %d. Spin-up steps: %d, Output every %d steps. Total: %d output frames. Diagnostics every %d steps, max writes per file: %d", 
            nsteps, spinup_step, output_freq, ((nsteps - spinup_step) / output_freq), diags_freq, Parameters.max_writes))
    
    println(@sprintf("Total time: %f, Time step: %f, Estimated CFL: %0.3f", Parameters.T, dt, Parameters.cfltune))

    true_output_dt = output_freq * dt
    true_sample_T = (nsteps - spinup_step) * dt
    println(@sprintf("(Nyquist freq/f, Rayleigh freq/f) = (%f, %f)", π/(true_output_dt)/Parameters.f, 2π/true_sample_T/Parameters.f))
    
    prob = RotatingShallowWater.Problem(dev; Lx, nx, dt, f=Parameters.f, Cg=Parameters.Cg, T=Float32, nν, ν, aliased_fraction=Parameters.aliased_fraction, order=Parameters.filter_order, use_filter=Parameters.use_filter)
    grid, clock, vars, params = prob.grid, prob.clock, prob.vars, prob.params

    if Parameters.random_initial_condition
        set_shafer_initial_condition!(prob, Parameters.Kg, Parameters.Kw, Parameters.ag, Parameters.aw, params.f, params.Cg2)
    else
        load_initial_condition_from_file!(prob, Parameters.snapshot_file, Parameters.snapshot_key)
    end

    return prob, nsteps, spinup_step, output_freq, diags_freq
end

function savediagnostics(diagnostics, diagnostic_names)
    for (i, diag) in enumerate(diagnostics)
        savediagnostic(diag, diagnostic_names[i], "diagnostics.jld2")
    end
end

function start!()
    prob, nsteps, spinup_step, output_freq, diags_freq = initialize_problem()
    grid, clock, vars, params = prob.grid, prob.clock, prob.vars, prob.params
    
    filename_func(idx) = @sprintf("%s.%06d.jld2", Parameters.base_filename, idx)

    get_sol(prob) = Array(prob.sol)
    output = SequencedOutput(prob, filename_func, (:sol, get_sol), Parameters.max_writes)

    rsw_KE = Diagnostic(kinetic_energy, prob; nsteps, freq=diags_freq)
    rsw_PE = Diagnostic(potential_energy, prob; nsteps, freq=diags_freq)
    diags = [rsw_KE, rsw_PE]
    diag_names = ["KE", "PE"]

    RotatingShallowWater.enforce_reality_condition!(prob)
    SequencedOutputs.saveproblem(output)
    SequencedOutputs.saveoutput(output)
    
    frames = 0:round(Int, nsteps / output_freq)

    starttime = time()
    for step=frames
        if (step % 1000  == 0)
            max_udx = max(maximum(abs.(vars.u)) / grid.dx, maximum(abs.(vars.v)) / grid.dy)
            cfl = clock.dt * max_udx
            println(@sprintf("step: %04d, t: %.2f, cfl: %.2e, time: %.2f mins", clock.step, clock.t, cfl, (time() - starttime) / 60))
            flush(stdout)
        end
        RotatingShallowWater.stepforward!(prob, diags, output_freq)
        if(any(isnan.(vars.η)))
            println("Blew up at step ", clock.step)
            savediagnostics(diags, diag_names)
            SequencedOutputs.close(output)
            throw("Solution is NaN")
        end
        RotatingShallowWater.updatevars!(prob)
        if (clock.step >= spinup_step)
            SequencedOutputs.saveoutput(output)
        end
    end
    savediagnostics(diags, diag_names)
    SequencedOutputs.close(output)
end
