using .RotatingShallowWater
using .SequencedOutputs
using FourierFlows
using Printf
using LinearAlgebra: mul!, ldiv!

import .Parameters

function set_shafer_initial_condition!(prob, Kg, Kw, ag, aw, f, Cg2)
    grid = prob.grid
    dev = typeof(grid.device)
    T = typeof(grid.Lx)

    @devzeros dev Complex{T} (grid.nkr, grid.nl) ugh vgh ηgh uwh vwh ηwh
    @devzeros dev T (grid.nx, grid.ny) ug uw
    
    geo_filter  = Kg[1]^2 .<= grid.Krsq .<= Kg[2]^2
    wave_filter = Kw[1]^2 .<= grid.Krsq .<= Kw[2]^2
    phase = device_array(grid.device)(2π*rand(grid.nkr, grid.nl))
    sgn =  device_array(grid.device)(sign.(rand(grid.nkr, grid.nl) .- 0.5))
    shift = exp.(1im * phase)
    ηgh[geo_filter] += ( 0.5   * shift)[geo_filter]
    ugh[geo_filter] += (-0.5im * Cg2 * f * grid.l  .* shift)[geo_filter]
    vgh[geo_filter] += ( 0.5im * Cg2 * f * grid.kr .* shift)[geo_filter]

    ldiv!(ug, grid.rfftplan, deepcopy(ugh))
    ηgh *= ag / maximum(abs.(ug))
    ugh *= ag / maximum(abs.(ug))
    vgh *= ag / maximum(abs.(ug))

    ωK =  sgn .* sqrt.(f^2 .+ Cg2 * grid.Krsq)
    ηwh[wave_filter] += (0.5 * shift)[wave_filter]
    uwh[wave_filter] += (grid.invKrsq.*(0.5 * grid.kr .* ωK .* shift + 0.5im * f * grid.l  .* shift))[wave_filter]
    vwh[wave_filter] += (grid.invKrsq.*(0.5 * grid.l .*  ωK .* shift - 0.5im * f * grid.kr .* shift))[wave_filter]
    
    ldiv!(uw, grid.rfftplan, deepcopy(uwh))
    ηwh *= aw / maximum(abs.(uw))
    uwh *= aw / maximum(abs.(uw))
    vwh *= aw / maximum(abs.(uw))
    RotatingShallowWater.set_solution!(prob, ugh + uwh, vgh + vwh, ηgh + ηwh)
end

function initialize_problem()
    dev = GPU()
    Lx=Parameters.L
    nx=Parameters.nx
    dx=Lx/nx
    kmax = (nx/2 - 1) * Lx / (2π)
    nν=Parameters.nν

    umax = Parameters.ag + Parameters.aw
    νtune = Parameters.νtune
    dt = Parameters.cfltune / umax * dx
    ν = Parameters.νtune * dx / (kmax^(2*nν)) / dt
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
    
    set_shafer_initial_condition!(prob, Parameters.Kg, Parameters.Kw, Parameters.ag, Parameters.aw, params.f, params.Cg2)

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

    rswE = Diagnostic(energy, prob; nsteps, freq=diags_freq)
    diags = [rswE]
    diag_names = ["energy"]

    RotatingShallowWater.enforce_reality_condition!(prob)
    SequencedOutputs.saveproblem(output)
    SequencedOutputs.saveoutput(output)
    
    frames = 0:round(Int, nsteps / output_freq)

    starttime = time()
    for step=frames
        if (step % (1000 / output_freq)  == 0)
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
