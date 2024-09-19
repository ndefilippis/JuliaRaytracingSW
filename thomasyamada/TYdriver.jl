module Driver

include("ThomasYamada.jl")
include("TYUtils.jl")
include("Parameters.jl")

using FourierFlows
using FourierFlows: parsevalsum2

using Random: seed!
using Printf
using JLD2
using .ThomasYamada
using .TYUtils
import .Parameters

function kinetic_energy_spectrum(solution, grid)
    Gh, Wh = decompose_balanced_wave(solution, grid)
    KEth = solution[:,:,1].^2
    KEgh = Gh[:,:,1].^2 + Gh[:,:,2].^2
    KEwh = Wh[:,:,1].^2 + Wh[:,:,2].^2
    KEtr = FourierFlows.radialspectrum(KEth, grid, refinement = 1)  
    KEgr = FourierFlows.radialspectrum(KEgh, grid, refinement = 1)
    KEwr = FourierFlows.radialspectrum(KEwh, grid, refinement = 1)
    return (KEtr, KEgr, KEwr)
end

function set_initial_condition_from_file(prob, restart_file, snapshot_key)
    file = jldopen(restart_file)
    sol = file[snapshot_key]
    close(file)
    set_solution!(prob, sol[:,:,1], sol[:,:,2], sol[:,:,3], sol[:,:,4])
end

function set_initial_condition(prob; k0w_range=(0, 1), k0g_range=(0, 1), at=0.0, ag=0.0, aw=0.0)
    grid = prob.grid
    dev = grid.device
    seed!(5678)
    
	k0w_min, k0w_max = k0w_range
	k0g_min, k0g_max = k0g_range
    wave_filter = (k0w_min^2 .<= grid.Krsq .<= k0w_max^2)
	geo_filter  = (k0g_min^2 .<= grid.Krsq .<= k0g_max^2)
    
    θ  = device_array(dev)(rand(Float64, (grid.nkr, grid.nl)))
    θ₀ = device_array(dev)(rand(Float64, (grid.nkr, grid.nl)))
    θ₊ = device_array(dev)(rand(Float64, (grid.nkr, grid.nl)))
    θ₋ = device_array(dev)(rand(Float64, (grid.nkr, grid.nl)))
    ϕ  = @. exp(2*π*im*θ )
    ϕ₀ = @. exp(2*π*im*θ₀)
    ϕ₊ = @. exp(2*π*im*θ₊)
    ϕ₋ = @. exp(2*π*im*θ₋)
    
    Φ₀ = compute_balanced_basis(grid)
    Φ₊, Φ₋ = compute_wave_bases(grid)
    
    ψth = ϕ .* geo_filter
    
    ugh = Φ₀[:,:,1] .* ϕ₀ .* geo_filter
    vgh = Φ₀[:,:,2] .* ϕ₀ .* geo_filter
    pgh = Φ₀[:,:,3] .* ϕ₀ .* geo_filter
    
    uwh = (Φ₊[:,:,1] .* ϕ₊ + Φ₋[:,:,1] .* ϕ₋) .* wave_filter
    vwh = (Φ₊[:,:,2] .* ϕ₊ + Φ₋[:,:,2] .* ϕ₋) .* wave_filter
    pwh = (Φ₊[:,:,3] .* ϕ₊ + Φ₋[:,:,3] .* ϕ₋) .* wave_filter

    ψt = irfft(ψth, grid.nx, (1, 2))
    pg = irfft(pgh, grid.nx, (1, 2))
    pw = irfft(pwh, grid.nx, (1, 2))
    
    max_ψt = maximum(abs.(ψt))
    max_pg = maximum(abs.(pg))
    max_pw = maximum(abs.(pw))

    @. ψth = ψth * at / max_ψt
    
    @. ugh = ugh * ag / max_pg
    @. vgh = vgh * ag / max_pg
    @. pgh = pgh * ag / max_pg
    
    @. uwh = uwh * aw / max_pw
    @. vwh = vwh * aw / max_pw
    @. pwh = pwh * aw / max_pw

    ζ₀h = @. - grid.Krsq * ψth
    u₀h = uwh + ugh
    v₀h = vwh + vgh
    p₀h = pwh + pgh
    
    set_solution!(prob, ζ₀h, u₀h, v₀h, p₀h)
end

function create_figure(prob, ζt, qc, sol, baroclinic, barotropic)
    fig = Figure(size=(1200,400))
    axζ = Axis(fig[1,1][1,1]; title="ζ_T")
    axq = Axis(fig[1,2][1,1]; title="q_C")
    axE = Axis(fig[1,3][1,1], xlabel="t", ylabel="bc E", limits=((0, Parameters.nsteps * prob.clock.dt), (0, 10.)))

    ζhm = heatmap!(axζ, prob.grid.x, prob.grid.y, ζt; colormap = :balance)
    qhm = heatmap!(axq, prob.grid.x, prob.grid.y, qc; colormap = :balance)

    Colorbar(fig[1,1][2, 1], ζhm, vertical=false, flipaxis = false)
    Colorbar(fig[1,2][2, 1], qhm, vertical=false, flipaxis = false)
    bc = lines!(axE, baroclinic, label="bc E"; linewidth = 3)
    bt = lines!(axE, barotropic, label="bt E"; linewidth = 3)
    axislegend(axE) 
    
    return fig
end

function start!()
    startup_nsteps = Parameters.startup_nsteps
    startup_nsubs = Parameters.startup_nsubs
    nsteps = Parameters.nsteps
    nsubs = Parameters.nsubs
   
    dev = CPU()
    if (ARGS[1] == "GPU")
		println("Executing on the GPU...")
        dev = GPU() 
    end
    startup_prob = Problem(dev;
		Lx = Parameters.Lx, 
        nx = Parameters.nx,
        ν  = Parameters.ν,
        nν = Parameters.nν,
        Ro = Parameters.Ro,
        stepper = Parameters.stepper,
        dt = Parameters.startup_dt)
    
    sol, clock, params, vars, grid = startup_prob.sol, startup_prob.clock, startup_prob.params, startup_prob.vars, startup_prob.grid
    x, y = grid.x, grid.y

    if (length(ARGS) >= 2)
        set_initial_condition_from_file(startup_prob, Parameters.restart_file, Parameters.restart_frame)
    else
        set_initial_condition(startup_prob; k0g_range=Parameters.k0g_range, k0w_range=Parameters.k0w_range, 
            at=Parameters.at, ag=Parameters.ag, aw=Parameters.aw)
    end

    filepath = "."
    file_index = 0
    base_filename = joinpath(filepath, Parameters.filename)
    get_filename(file_index) = @sprintf("%s.%08d", base_filename, file_index)
    max_writes = Parameters.max_writes

    filename = get_filename(file_index)
    if isfile(filename); rm(filename); end
    
    get_sol(prob) = Array(prob.sol)
    
    wave_geo_E = Diagnostic(wave_geostrophic_energy, startup_prob; nsteps=startup_nsteps, freq=25)
    btE = Diagnostic(barotropic_energy, startup_prob; nsteps=startup_nsteps, freq=25)
    diags = [wave_geo_E, btE]

    out = Output(startup_prob, "startup", (:sol, get_sol))
    startwalltime = time()
    updatevars!(startup_prob)
    saveproblem(out)
    saveoutput(out)
    current_writes = 1

    startup_frames=0:round(Int, startup_nsteps / startup_nsubs)
    for j=startup_frames
        if j % (4000 / startup_nsubs) == 0
            max_udx = maximum([maximum(vars.uc) / grid.dx, maximum(vars.vc) / grid.dy, maximum(vars.ut) / grid.dx, maximum(vars.vt) / grid.dy])
            cfl = clock.dt * max_udx
            log = @sprintf("step %04d, t:%.1f, cfl: %.4f, walltime: %.2f min", clock.step, clock.t, cfl, (time()-startwalltime)/60)
            println(log)
			flush(stdout)
        end
        stepforward!(startup_prob, diags, startup_nsubs)
        enforce_reality_condition!(startup_prob)
        updatevars!(startup_prob)
    end
    savediagnostic(btE, "barotropic_energy", "startup_diagnostics.jld2")
    savediagnostic(wave_geo_E, "wave_geostrophic_energy", "startup_diagnostics.jld2")
    saveoutput(out)
    println("Startup finished")

    prob = Problem(CPU();
		Lx = Parameters.Lx, 
        nx = Parameters.nx,
        ν  = Parameters.ν,
        nν = Parameters.nν,
        Ro = Parameters.Ro,
        stepper = Parameters.stepper,
        dt = Parameters.dt)

    prob.clock.t = startup_prob.clock.t
    set_solution!(prob, startup_prob.sol[:,:,1], startup_prob.sol[:,:,2], startup_prob.sol[:,:,3], startup_prob.sol[:,:,4])

    startup_prob = nothing
    wave_geo_E = Diagnostic(wave_geostrophic_energy, prob; nsteps=startup_nsteps, freq=25)
    btE = Diagnostic(barotropic_energy, prob; nsteps=startup_nsteps, freq=25)
    diags = [wave_geo_E, btE]
    
    sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
    x, y = grid.x, grid.y

    out = Output(prob, filename, (:sol, get_sol))
    updatevars!(prob)
    saveproblem(out)
    saveoutput(out)

    frames = 0:round(Int, nsteps / nsubs)
    
    for j=frames
    	# record(fig, "thomas_yamada.mp4", frames, framerate = 18) do j
        if j % (1000 / nsubs) == 0
            max_udx = maximum([maximum(vars.uc) / grid.dx, maximum(vars.vc) / grid.dy, maximum(vars.ut) / grid.dx, maximum(vars.vt) / grid.dy])
            cfl = clock.dt * max_udx
            log = @sprintf("step %04d, t:%.1f, cfl: %.4f, walltime: %.2f min", clock.step, clock.t, cfl, (time()-startwalltime)/60)
            println(log)
			flush(stdout)	
        end
        stepforward!(prob, diags, nsubs)
        enforce_reality_condition!(prob)
        updatevars!(prob)
        saveoutput(out)
        current_writes += 1
        if current_writes >= max_writes
            current_writes = 0
            file_index += 1
            filename = get_filename(file_index)
            out = Output(prob, filename, (:sol, get_sol))
        end
    end
    savediagnostic(btE, "barotropic_energy", "diagnostics.jld2")
    savediagnostic(wave_geo_E, "baroclinic_energy", "diagnostics.jld2")
end

start!()
end
