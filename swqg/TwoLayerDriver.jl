using .TwoLayerQG
using .SequencedOutputs
using FourierFlows
using Printf
using Random
using LinearAlgebra: ldiv!

import .Parameters

function set_seed_initial_condition!(prob, grid, dev)
    q0 = 1e-2 * device_array(dev)(randn((grid.nx, grid.ny, 2)))
    q0h = rfft(q0, (1, 2))

    TwoLayerQG.set_solution!(prob, q0h)
end

function compute_parameters(deformation_radius, intervortex_radius, avg_eddy_velocity, H, f0)
    c₁ = 3.2
    c₂ = 0.36
    l_star = intervortex_radius/deformation_radius

    kappa_star = c₂/log(l_star/c₁)
    U = avg_eddy_velocity / l_star
    μ = 2*U*kappa_star/deformation_radius; # bottom drag
    δb = 4 * f0^2 * deformation_radius^2/H
    return μ, δb, U
end

function initialize_problem()
    dev = GPU()
    Lx=Parameters.L
    nx=Parameters.nx
    dx=Lx/nx
    kmax = (nx/2 - 1) * (1 - Parameters.aliased_fraction)
    nν=Parameters.nν
    νtune = Parameters.νtune
    H = 1.0
    
    μ, δb, U = compute_parameters(Parameters.deformation_radius, Parameters.intervortex_radius, Parameters.ug, H, Parameters.f)
    δρρ0 = δb / (Parameters.background_Cg/H)
    
    if(μ < 0)
        println("Exiting: μ < 0: ", μ)
        return nothing
    end
    
    cfltune = Parameters.cfltune
    umax = Parameters.ug
    
    dt = cfltune / umax * dx
    ν = νtune * 2π / nx / (kmax^(2*nν)) / dt
    
    nsteps = ceil(Int, Parameters.T / dt)
    spinup_step = floor(Int, Parameters.spinup_T / dt)
    output_freq = max(floor(Int, Parameters.output_dt / dt), 1)
    diags_freq = max(floor(Int, Parameters.diag_dt / dt), 1)

    println(@sprintf("Total steps: %d. Spin-up steps: %d, Output every %d steps. Total: %d output frames. Diagnostics every %d steps, max writes per file: %d", 
            nsteps, spinup_step, output_freq, ((nsteps - spinup_step) / output_freq), diags_freq, Parameters.max_writes))
    
    println(@sprintf("Total time: %f, Time step: %f, Estimated CFL: %0.3f", Parameters.T, dt, Parameters.cfltune))
    
    prob = TwoLayerQG.Problem(dev; stepper="IFMAB3", Lx, nx, dt, f0=Parameters.f, Cg=Parameters.background_Cg, U, δρρ0, T=Float32, nν, ν, μ, aliased_fraction=Parameters.aliased_fraction, use_filter=false)
    
    set_seed_initial_condition!(prob, prob.grid, dev)

    return prob, nsteps, spinup_step, output_freq, diags_freq
end

function savediagnostics(diagnostics, diagnostic_names)
    for (i, diag) in enumerate(diagnostics)
        savediagnostic(diag, diagnostic_names[i], "diagnostics.jld2")
    end
end

function start!()
    Random.seed!(1234)
    prob, nsteps, spinup_step, output_freq, diags_freq = initialize_problem()
    grid, clock, vars, params = prob.grid, prob.clock, prob.vars, prob.params
    
    filename_func(idx) = @sprintf("%s.%06d.jld2", Parameters.base_filename, idx)

    get_sol(prob) = Array(prob.sol)
    output = SequencedOutput(prob, filename_func, (:sol, get_sol), Parameters.max_writes)

    KEdiag  = Diagnostic(kinetic_energy, prob; nsteps, freq=diags_freq)
    PEdiag  = Diagnostic(potential_energy, prob; nsteps, freq=diags_freq)
    diags = [KEdiag, PEdiag]
    diag_names = ["kinetic_energy", "potential_energy"]

    TwoLayerQG.enforce_reality_condition!(prob)
    SequencedOutputs.saveproblem(output)
    SequencedOutputs.saveoutput(output)
    
    frames = 0:round(Int, nsteps / output_freq)

    starttime = time()
    for step=frames
        if (step % 100 == 0)
            max_udx = max(maximum(abs.(vars.u)) / grid.dx, maximum(abs.(vars.v)) / grid.dy)
            cfl = clock.dt * max_udx
            println(@sprintf("step: %04d, t: %.2f, cfl: %.2e, time: %.2f mins", clock.step, clock.t, cfl, (time() - starttime) / 60))
            flush(stdout)
        end
        TwoLayerQG.stepforward!(prob, diags, output_freq)
        if(any(isnan.(vars.q)))
            println("Blew up at step ", clock.step)
            savediagnostics(diags, diag_names)
            SequencedOutputs.close(output)
            throw("Solution is NaN")
        end
        TwoLayerQG.updatevars!(prob)
        if (clock.step >= spinup_step)
            SequencedOutputs.saveoutput(output)
        end
    end
    savediagnostics(diags, diag_names)
    SequencedOutputs.close(output)
end
