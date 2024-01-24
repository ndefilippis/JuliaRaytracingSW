module Driver

using FourierFlows
using Random: seed!
using CairoMakie
using Printf
using .ThomasYamada
import .Parameters

function compute_balanced_basis(grid)
    Φ₀ = Array{Complex{Float64}, 3}(undef, grid.nkr, grid.nl, 3)
    ω = @. sqrt(1 + grid.kr^2 + grid.l^2)
    @. Φ₀[:,:,1] =  im * grid.l  / ω
    @. Φ₀[:,:,2] = -im * grid.kr / ω
    @. Φ₀[:,:,3] = -1 / ω
    @. Φ₀[1,1,:] = [0, 0, 1]
    
    return Φ₀
end

function compute_wave_bases(grid)
    ω = @. sqrt(1 + grid.kr^2 + grid.l^2)
    Φ₊ = Array{Complex{Float64}, 3}(undef, grid.nkr, grid.nl, 3)
    Φ₋ = Array{Complex{Float64}, 3}(undef, grid.nkr, grid.nl, 3)
    
    @. Φ₊[:,:,1] = (ω*grid.kr + im * grid.l) * sqrt(grid.invKrsq/2)/ω
    @. Φ₊[:,:,2] = (ω*grid.l - im * grid.kr) * sqrt(grid.invKrsq/2)/ω
    @. Φ₊[:,:,3] = (ω^2 - 1) * sqrt(grid.invKrsq/2)/ω
    @. Φ₊[1,1,:] = [im, 1, 0]/sqrt(2)
    
    @. Φ₋[:,:,1] = (-ω*grid.kr + im * grid.l) * sqrt(grid.invKrsq/2)/ω
    @. Φ₋[:,:,2] = (-ω*grid.l - im * grid.kr) * sqrt(grid.invKrsq/2)/ω
    @. Φ₋[:,:,3] = (ω^2 - 1) * sqrt(grid.invKrsq/2)/ω
    @. Φ₋[1,1,:] = [im, -1, 0]/sqrt(2)
   
    return (Φ₊, Φ₋)
end

function set_initial_condition(prob; k0=0, Et=0.0, Eg=0.0, Ew=0.0)
    grid = prob.grid
    dev = grid.device
    seed!(5678)
    
    filter = (grid.Krsq .<= k0^2)
    
    θ = device_array(dev)(rand(Float64, (grid.nkr, grid.nl)))
    ψth = @. exp(2*pi*im*θ)*filter
    ut = irfft(-im * grid.l  .* ψth, grid.nx, (1, 2))
    vt = irfft( im * grid.kr .* ψth, grid.nx, (1, 2))
    btE = sum(ut.^2 + vt.^2) * grid.dx * grid.dy
    @. ψth = ψth * sqrt(Et / btE)
    ζ₀h = @. - grid.Krsq * ψth
    ζ₀ = irfft(ζ₀h, grid.nx, (1, 2))
    
    θ₀ = device_array(dev)(rand(Float64, (grid.nkr, grid.nl)))
    θ₊ = device_array(dev)(rand(Float64, (grid.nkr, grid.nl)))
    θ₋ = device_array(dev)(rand(Float64, (grid.nkr, grid.nl)))
    ϕ₀ = @. exp(2*π*im*θ₀) * filter
    ϕ₊ = @. exp(2*π*im*θ₊) * filter
    ϕ₋ = @. exp(2*π*im*θ₋) * filter
    ω = @. sqrt(1 + grid.Krsq)
    
    ugh = @.  ϕ₀*im*grid.l/ω
    vgh = @. -ϕ₀*im*grid.kr/ω
    pgh = @. -ϕ₀/ω
    ugh[1,1] = 0
    vgh[1,1] = 0
    pgh[1,1] = ϕ₀[1,1]
    
    factor = @. sqrt(grid.invKrsq/2)/ω
    uwh = @. factor * ((ω*grid.kr + im*grid.l)  * ϕ₊ + (-ω*grid.kr + im*grid.l)  * ϕ₋)
    vwh = @. factor * ((ω*grid.l  - im*grid.kr) * ϕ₊ + (-ω*grid.l  - im*grid.kr) * ϕ₋)
    pwh = @. factor * (ω^2 - 1) * (ϕ₊ + ϕ₋)
    uwh[1,1] = im/sqrt(2)*(ϕ₊[1,1] + ϕ₋[1,1])
    vwh[1,1] =  1/sqrt(2)*(ϕ₊[1,1] - ϕ₋[1,1])
    pwh[1,1] =  0
    
    ug = irfft(ugh, grid.nx, (1, 2))
    vg = irfft(vgh, grid.nx, (1, 2))
    pg = irfft(pgh, grid.nx, (1, 2))
    
    gE = sum(@. ug^2 + vg^2 + pg^2) * grid.dx * grid.dy
    @. ug = ug * sqrt(Eg / gE)
    @. vg = vg * sqrt(Eg / gE)
    @. pg = pg * sqrt(Eg / gE)
    
    uw = irfft(uwh, grid.nx, (1, 2))
    vw = irfft(vwh, grid.nx, (1, 2))
    pw = irfft(pwh, grid.nx, (1, 2))
    
    wE = sum(@. uw^2 + vw^2 + pw^2) * grid.dx * grid.dy
    @. uw = uw * sqrt(Ew / wE)
    @. vw = vw * sqrt(Ew / wE)
    @. pw = pw * sqrt(Ew / wE)
    
    u₀ = uw + ug
    v₀ = vw + vg
    p₀ = pw + pg
    
    set_solution!(prob, ζ₀, u₀, v₀, p₀)
end

function start!()
    nsteps = Parameters.nsteps
    nsubs = Parameters.nsubs
    sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
    x, y = grid.x, grid.y
    
    dev = CPU()
    prob = Problem(dev; 
        nx = Parameters.nx,
        ν  = Parameters.ν,
        nν = Parameters.nν,
        Ro = Parameters.Ro,
        stepper = Parameters.stepper,
        dt = Parameters.dt)

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
    baroclinic = Observable(Point2f[(bcE.t[1], bcE.data[1])])
    barotropic = Observable(Point2f[(btE.t[1], btE.data[1])])

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


    startwalltime = time()
    frames = 0:round(Int, nsteps / nsubs)

    updatevars!(prob)
    saveproblem(out)
    record(fig, "thomas_yamada.mp4", frames, framerate = 18) do j
        if j % (1000 / nsubs) == 0
            max_udx = maximum([maximum(vars.uc) / grid.dx, maximum(vars.vc) / grid.dy, maximum(vars.ut) / grid.dx, maximum(vars.vt) / grid.dy])
            cfl = clock.dt * max_udx
            log = @sprintf("step %04d, t:%.1f, cfl: %.4f, walltime: %.2f min", clock.step, clock.t, cfl, (time()-startwalltime)/60)
            println(log)
        end
        ζt[] = vars.ζt
        qc[] = vars.qc
        baroclinic[] = push!(baroclinic[], Point2f(bcE.t[bcE.i], bcE.data[bcE.i][1]))
        barotropic[] = push!(barotropic[], Point2f(btE.t[btE.i], btE.data[btE.i][1]))
        stepforward!(prob, diags, nsubs)
        updatevars!(prob)
        saveoutput(out)
    end
end

start!()
end