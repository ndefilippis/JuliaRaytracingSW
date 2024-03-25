module Driver

include("ThomasYamada.jl")
include("TYUtils.jl")
include("Parameters.jl")

using FourierFlows
using FourierFlows: parsevalsum2

using Random: seed!
using CairoMakie
using Printf
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

function set_initial_condition(prob; k0w_range=(0, 1), k0g_range=(0, 1), Et=0.0, Eg=0.0, Ew=0.0)
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
    
    btE = parsevalsum2(sqrt.(grid.Krsq) .* ψth, grid)
    
    gE  = parsevalsum2(ugh, grid) + parsevalsum2(vgh, grid) + parsevalsum2(pgh, grid)
    wE  = parsevalsum2(uwh, grid) + parsevalsum2(vwh, grid) + parsevalsum2(pwh, grid)
    
    @. ψth = ψth * sqrt(Et / btE)
    
    
    @. ugh = ugh * sqrt(Eg / gE)
    @. vgh = vgh * sqrt(Eg / gE)
    @. pgh = pgh * sqrt(Eg / gE)
    
    @. uwh = uwh * sqrt(Ew / wE)
    @. vwh = vwh * sqrt(Ew / wE)
    @. pwh = pwh * sqrt(Ew / wE)

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
    nsteps = Parameters.nsteps
    nsubs = Parameters.nsubs
   
    dev = CPU()
    if (ARGS[1] == "GPU")
		println("Executing on the GPU...")
        dev = GPU() 
    end
    prob = Problem(dev;
		Lx = Parameters.Lx, 
        nx = Parameters.nx,
        ν  = Parameters.ν,
        nν = Parameters.nν,
        Ro = Parameters.Ro,
        stepper = Parameters.stepper,
        dt = Parameters.dt)
    
    sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
    x, y = grid.x, grid.y

    set_initial_condition(prob; k0g_range=Parameters.k0g_range, k0w_range=Parameters.k0w_range, Et=Parameters.Et, Eg=Parameters.Eg, Ew=Parameters.Ew)

    filepath = "."
    filename = joinpath(filepath, Parameters.filename)
    if isfile(filename); rm(filename); end
    
    get_sol(prob) = Array(prob.sol)
    out = Output(prob, filename, (:sol, get_sol))
    
    bcE = Diagnostic(baroclinic_energy, prob; nsteps)
    btE = Diagnostic(barotropic_energy, prob; nsteps)
    diags = [bcE, btE]

    # ζt = Observable(Array(vars.ζt))
    # qc = Observable(Array(vars.qc))
    # solution = Observable(Array(sol))
    # baroclinic = Observable(Point2f[(bcE.t[1], bcE.data[1])])
    # barotropic = Observable(Point2f[(btE.t[1], btE.data[1])])

    # fig = create_figure(prob, ζt, qc, sol, baroclinic, barotropic)

    startwalltime = time()
    frames = 0:round(Int, nsteps / nsubs)

    updatevars!(prob)
    saveproblem(out)
    saveoutput(out)
    for j=frames
    	# record(fig, "thomas_yamada.mp4", frames, framerate = 18) do j
        if j % (1000 / nsubs) == 0
            max_udx = maximum([maximum(vars.uc) / grid.dx, maximum(vars.vc) / grid.dy, maximum(vars.ut) / grid.dx, maximum(vars.vt) / grid.dy])
            cfl = clock.dt * max_udx
            log = @sprintf("step %04d, t:%.1f, cfl: %.4f, walltime: %.2f min", clock.step, clock.t, cfl, (time()-startwalltime)/60)
            println(log)
			flush(stdout)	
        end
        # ζt[] = vars.ζt
        # qc[] = vars.qc
        # baroclinic[] = push!(baroclinic[], Point2f(bcE.t[bcE.i], bcE.data[bcE.i][1]))
        # barotropic[] = push!(barotropic[], Point2f(btE.t[btE.i], btE.data[btE.i][1]))
        # solution[] = sol
        stepforward!(prob, diags, nsubs)
        updatevars!(prob)
        saveoutput(out)
    end
end

start!()
end
