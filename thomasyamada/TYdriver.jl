module Driver

include("ThomasYamada.jl")
include("Parameters.jl")

using FourierFlows
using FourierFlows: parsevalsum2

using Random: seed!
using CairoMakie
using Printf
using .ThomasYamada
import .Parameters

function compute_balanced_basis(grid)
    Φ₀ = device_array(grid.device)(Array{Complex{Float64}, 3}(undef, grid.nkr, grid.nl, 3))
    ω = @. sqrt(1 + grid.kr^2 + grid.l^2)
    @. Φ₀[:,:,1] =  im * grid.l  / ω
    @. Φ₀[:,:,2] = -im * grid.kr / ω
    @. Φ₀[:,:,3] = -1 / ω
    Φ₀[1,1,:] = device_array(grid.device)([0, 0, 1])
    
    return Φ₀
end

function compute_wave_bases(grid)
    ω = @. sqrt(1 + grid.kr^2 + grid.l^2)
    Φ₊ = device_array(grid.device)(Array{Complex{Float64}, 3}(undef, grid.nkr, grid.nl, 3))
    Φ₋ = device_array(grid.device)(Array{Complex{Float64}, 3}(undef, grid.nkr, grid.nl, 3))
    
    @. Φ₊[:,:,1] = (ω*grid.kr + im * grid.l) * sqrt(grid.invKrsq/2)/ω
    @. Φ₊[:,:,2] = (ω*grid.l - im * grid.kr) * sqrt(grid.invKrsq/2)/ω
    @. Φ₊[:,:,3] = (ω^2 - 1) * sqrt(grid.invKrsq/2)/ω
    Φ₊[1,1,:] = device_array(grid.device)([im, 1, 0]/sqrt(2))
    
    @. Φ₋[:,:,1] = (-ω*grid.kr + im * grid.l) * sqrt(grid.invKrsq/2)/ω
    @. Φ₋[:,:,2] = (-ω*grid.l - im * grid.kr) * sqrt(grid.invKrsq/2)/ω
    @. Φ₋[:,:,3] = (ω^2 - 1) * sqrt(grid.invKrsq/2)/ω
    Φ₋[1,1,:] = device_array(grid.device)([im, -1, 0]/sqrt(2))
   
    return (Φ₊, Φ₋)
end

function decompose_balanced_wave(solution, grid)
    Φ₀ = compute_balanced_basis(prob.grid)
    baroclinic_components = solution[:,:,2:4]
    Gh = sum(baroclinic_components .* conj(Φ₀), dims=3) .* Φ₀
    Wh = baroclinic_components - Gh
    return (G, W)
end

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

function set_initial_condition(prob; k0=0, Et=0.0, Eg=0.0, Ew=0.0)
    grid = prob.grid
    dev = grid.device
    seed!(5678)
    
    filter = (grid.Krsq .<= k0^2)
    
    θ  = device_array(dev)(rand(Float64, (grid.nkr, grid.nl)))
    θ₀ = device_array(dev)(rand(Float64, (grid.nkr, grid.nl)))
    θ₊ = device_array(dev)(rand(Float64, (grid.nkr, grid.nl)))
    θ₋ = device_array(dev)(rand(Float64, (grid.nkr, grid.nl)))
    ϕ  = @. exp(2*π*im*θ ) * filter
    ϕ₀ = @. exp(2*π*im*θ₀) * filter
    ϕ₊ = @. exp(2*π*im*θ₊) * filter
    ϕ₋ = @. exp(2*π*im*θ₋) * filter
    
    Φ₀ = compute_balanced_basis(grid)
    Φ₊, Φ₋ = compute_wave_bases(grid)
    
    ψth = ϕ
    
    ugh = Φ₀[:,:,1] .* ϕ₀
    vgh = Φ₀[:,:,2] .* ϕ₀
    pgh = Φ₀[:,:,3] .* ϕ₀
    
    uwh = Φ₊[:,:,1] .* ϕ₊ + Φ₋[:,:,1] .* ϕ₋
    vwh = Φ₊[:,:,2] .* ϕ₊ + Φ₋[:,:,2] .* ϕ₋
    pwh = Φ₊[:,:,3] .* ϕ₊ + Φ₋[:,:,3] .* ϕ₋
    
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

function create_figure(ζt, qc, sol, baroclinic, barotropic)
    fig = Figure(size=(1200,400))
    axζ = Axis(fig[1,1][1,1]; title="ζ_T")
    axq = Axis(fig[1,2][1,1]; title="q_C")
    axE = Axis(fig[1,3][1,1], xlabel="t", ylabel="bc E", limits=((0, nsteps * prob.clock.dt), (0, 10.)))

    ζhm = heatmap!(axζ, x, y, ζt; colormap = :balance)
    qhm = heatmap!(axq, x, y, qc; colormap = :balance)

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
    if (Parameters.device == "GPU")
       dev = GPU() 
    end
    prob = Problem(dev; 
        nx = Parameters.nx,
        ν  = Parameters.ν,
        nν = Parameters.nν,
        Ro = Parameters.Ro,
        stepper = Parameters.stepper,
        dt = Parameters.dt)
    
    sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
    x, y = grid.x, grid.y

    set_initial_condition(prob; k0=Parameters.k0, Et=Parameters.Et, Eg=Parameters.Eg, Ew=Parameters.Ew)

    filepath = "."
    filename = joinpath(filepath, Parameters.filename)
    if isfile(filename); rm(filename); end
    
    get_sol(prob) = prob.sol
    out = Output(prob, filename, (:sol, get_sol))
    
    bcE = Diagnostic(baroclinic_energy, prob; nsteps)
    btE = Diagnostic(barotropic_energy, prob; nsteps)
    diags = [bcE, btE]

    ζt = Observable(Array(vars.ζt))
    qc = Observable(Array(vars.qc))
    solution = Observable(Array(sol))
    baroclinic = Observable(Point2f[(bcE.t[1], bcE.data[1])])
    barotropic = Observable(Point2f[(btE.t[1], btE.data[1])])

    # fig = create_figure(ζt, qc, sol, baroclinic, barotropic)

    startwalltime = time()
    frames = 0:round(Int, nsteps / nsubs)

    updatevars!(prob)
    saveproblem(out)
    for j=frames
        #record(fig, "thomas_yamada.mp4", frames, framerate = 18) do j
        if j % (1000 / nsubs) == 0
            max_udx = maximum([maximum(vars.uc) / grid.dx, maximum(vars.vc) / grid.dy, maximum(vars.ut) / grid.dx, maximum(vars.vt) / grid.dy])
            cfl = clock.dt * max_udx
            log = @sprintf("step %04d, t:%.1f, cfl: %.4f, walltime: %.2f min", clock.step, clock.t, cfl, (time()-startwalltime)/60)
            println(log)
			flush(stdout)	
        end
        #ζt[] = vars.ζt
        #qc[] = vars.qc
        #baroclinic[] = push!(baroclinic[], Point2f(bcE.t[bcE.i], bcE.data[bcE.i][1]))
        #barotropic[] = push!(barotropic[], Point2f(btE.t[btE.i], btE.data[btE.i][1]))
        #solution[] = sol
        stepforward!(prob, diags, nsubs)
        updatevars!(prob)
        saveoutput(out)
    end
end

start!()
end
