using GeophysicalFlows, CairoMakie, Printf;
using Random: seed!
using FourierFlows: parsevalsum2

import .Parameters

function modal_energy(prob)
    Eh = prob.grid.Krsq.*abs2.(@views prob.vars.ψh[:,:,1])
    kr, Ehr = FourierFlows.radialspectrum(Eh, prob.grid)
    return Ehr
end


function start!()
    dev = Parameters.device
    nx = Parameters.nx
    stepper = Parameters.stepper  # timestepper
    nsteps = Parameters.nsteps # total number of time-steps
    nsubs  = Parameters.nsubs             # number of time-steps for plotting (nsteps must be multiple of nsubs)

    nlayers = 2              # number of layers
    f₀, b = Parameters.f, Parameters.b            # Coriolis parameter and gravitational constant
    H = Parameters.H        # the rest depths of each layer

    Lx = Parameters.Lx                 # domain size

    β = 0                    # the y-gradient of planetary PV

    U = Parameters.U       # the imposed mean zonal flow in each layer

	μ = Parameters.μ		# bottom drag parameter

    dx = Lx/nx;
    dt = Parameters.dt         # timestep
    println(@sprintf("bottom drag: %.5f, time step: %.4f, shear flow: %.4f", μ, dt, (U[1] - U[2])/2));


    prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx, f₀, H, b, U, μ, β,
                                dt, stepper, aliased_fraction=0)
    sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
    x, y = grid.x, grid.y
    
    k_filter = Parameters.k0_min^2 .<= grid.Krsq .< Parameters.k0_max^2

    θ_bt = device_array(dev)(rand(Float64, (grid.nkr, grid.nl)))
    θ_bc = device_array(dev)(rand(Float64, (grid.nkr, grid.nl)))
    ψh_bt = @. exp(2*π*im*θ_bt) * k_filter
    ψh_bc = @. exp(2*π*im*θ_bc) * k_filter
    
    ψh_bt *= sqrt(Parameters.bt_energy / parsevalsum2(sqrt.(grid.Krsq) .* ψh_bt, grid))
    ψh_bc *= sqrt(Parameters.bc_energy / parsevalsum2(sqrt.(grid.Krsq) .* ψh_bc, grid))
    
    prob.vars.ψh[:, :, 1] = ψh_bt + ψh_bc
    prob.vars.ψh[:, :, 2] = ψh_bt - ψh_bc
    MultiLayerQG.pvfromstreamfunction!(prob.sol, prob.vars.ψh, params, grid)
    MultiLayerQG.updatevars!(prob)

    # Create Diagnostics -- `energies` function is imported at the top.
    radialE = Diagnostic(modal_energy, prob; nsteps)
    E = Diagnostic(MultiLayerQG.energies, prob; nsteps)
    diags = [E]

    filepath = Parameters.filepath
    filename = joinpath(filepath, Parameters.output_filename)
    if isfile(filename); rm(filename); end

    get_streamfunc(prob) = Array(prob.vars.ψh)
    out = Output(prob, filename, (:ψh, get_streamfunc))

    Lx, Ly = grid.Lx, grid.Ly

    title_KE = Observable(@sprintf("μt = %.2f", μ * clock.t))
    q = Observable(Array(vars.q[:, :, 1]))
    KE = Observable(Point2f[(μ * E.t[1], E.data[1][1][1])])
    ψh = Observable(vars.ψh[:,:,1])

    Eh = @lift prob.grid.Krsq.*abs2.($ψh) # Fourier transform of energy density
    krEhr = @lift FourierFlows.radialspectrum($Eh, grid, refinement = 1)
    kr = @lift Array($krEhr[1])
    Ehr = @lift vec(abs.(Array($krEhr[2]))) .+ 1e-9

	qcolorrange = @lift (-maximum(abs.($q)), maximum(abs.($q)))

    fig = Figure(size=(1800, 600))

    axis_kwargs = (xlabel = "x",
                   ylabel = "y",
                   aspect = 1,
                   limits = ((-Lx/2, Lx/2), (-Ly/2, Ly/2)))

    axq = Axis(fig[1, 1]; title = "q", axis_kwargs...)
    axKE = Axis(fig[1, 2],
                xlabel = "μ t",
                ylabel = "KE",
                title = title_KE,
                yscale = log10,
                aspect = 1,
                limits = ((-0.1, dt * μ * nsteps), (1e-9, 1)))
    axKEspec = Axis(fig[1, 3],
                xlabel = L"k_r",
                ylabel = L"\int |\hat{E}| k_r \mathrm{d}k_\theta",
                xscale = log10,
                yscale = log10,
                title = "Radial energy spectrum",
                aspect = 1,
                limits = ((1.0, nx/2-1), (1e-1, 1)))
    @lift ylims!(axKE, 1e-9, max(1, 2*maximum($KE).data[2]))
    @lift ylims!(axKEspec, 1e-1, max(1, 2*maximum($Ehr)))

    heatmap!(axq, x, y, q; colormap = :balance, colorrange=qcolorrange)

    ke = lines!(axKE, KE; linewidth = 3)
    kespec = lines!(axKEspec, kr, Ehr; linewidth = 2)
    startwalltime = time()

    frames = 0:round(Int, nsteps / nsubs)

    saveproblem(out)
    # CairoMakie.record(fig, "movie.mp4", frames, framerate = 18) do j
	for j=frames
      if j % (1000 / nsubs) == 0
        cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
        u_max = maximum([maximum(abs.(vars.u)), maximum(abs.(vars.v))])

        log = @sprintf("step: %04d, t: %.1f, cfl: %.4f, KE₁: %.3e, u_max: %.5e, walltime: %.2f min",
                       clock.step, clock.t, cfl, E.data[E.i][1][1], u_max, (time()-startwalltime)/60)

        println(log)
        flush(stdout)
      end
      KE[] = push!(KE[], Point2f(μ * E.t[E.i], E.data[E.i][1][1]))
      q[] = @views vars.q[:, :, 1]
      ψh[] = @views vars.ψh[:,:,1]
      title_KE[] = @sprintf("μ t = %.2f", μ * clock.t)

      stepforward!(prob, diags, nsubs)
      MultiLayerQG.updatevars!(prob)
      saveoutput(out);
    end
    
    snapshot_filename = joinpath(filepath, Parameters.snapshot_filename)
    if isfile(snapshot_filename); rm(snapshot_filename); end
    
    snapshot_out = Output(prob, snapshot_filename, (:ψh, get_streamfunc))
    saveproblem(snapshot_out)
    saveoutput(snapshot_out)
end
