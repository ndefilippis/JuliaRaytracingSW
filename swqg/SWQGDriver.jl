using .SWQG
using .SequencedOutputs
using FourierFlows
using Printf
using Random
using LinearAlgebra: ldiv!

import .Parameters

function set_shafer_initial_condition_QG!(prob, Kg, ag)
    grid = prob.grid
    dev = typeof(grid.device)
    T = typeof(grid.Lx)

    @devzeros dev Complex{T} (grid.nkr, grid.nl) ψh
    @devzeros dev T (grid.nx, grid.ny) ψ u
    
    geo_filter  = Kg[1]^2 .<= grid.Krsq .<= Kg[2]^2
    phase = device_array(grid.device)(2π*rand(grid.nkr, grid.nl))
    shift = exp.(1im * phase)
    ψh[geo_filter] += 0.5*shift[geo_filter]

    ldiv!(ψ, grid.rfftplan, deepcopy(ψh))
    ldiv!(u, grid.rfftplan, -grid.l .* ψh)
    
    ψh *= ag / maximum(abs.(u))
    SWQG.set_solution!(prob, ψh)
end

function initialize_problem()
    dev = GPU()
    Lx=Parameters.L
    nx=Parameters.nx
    dx=Lx/nx
    kmax = nx/2 - 1
    nν=Parameters.nν
    νtune = Parameters.νtune
    
    cfltune = Parameters.cfltune
    umax = Parameters.ag
    
    dt = cfltune / umax * dx
    ν = νtune * 2π / nx / (kmax^(2*nν)) / dt
    
    nsteps = ceil(Int, Parameters.T / dt)
    spinup_step = floor(Int, Parameters.spinup_T / dt)
    output_freq = max(floor(Int, Parameters.output_dt / dt), 1)
    diags_freq = max(floor(Int, Parameters.diag_dt / dt), 1)

    println(@sprintf("Total steps: %d. Spin-up steps: %d, Output every %d steps. Total: %d output frames. Diagnostics every %d steps, max writes per file: %d", 
            nsteps, spinup_step, output_freq, ((nsteps - spinup_step) / output_freq), diags_freq, Parameters.max_writes))
    
    println(@sprintf("Total time: %f, Time step: %f, Estimated CFL: %0.3f", Parameters.T, dt, Parameters.cfltune))
    
    prob = SWQG.Problem(dev; Lx, nx, dt, f, T=Float32, nν, ν, aliased_fraction=1/3, make_filter=false)
    
    set_shafer_initial_condition_QG!(prob, (10, 13), 0.3)

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

    rswE = Diagnostic(energy, prob; nsteps, freq=diags_freq)
    diags = []
    diag_names = []

    SWQG.enforce_reality_condition!(prob)
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
        SWQG.stepforward!(prob, diags, output_freq)
        if(any(isnan.(vars.η)))
            println("Blew up at step ", clock.step)
            savediagnostics(diags, diag_names)
            SequencedOutputs.close(output)
            throw("Solution is NaN")
        end
        SWQG.updatevars!(prob)
        if (clock.step >= spinup_step)
            SequencedOutputs.saveoutput(output)
        end
    end
    savediagnostics(diags, diag_names)
    SequencedOutputs.close(output)
end
