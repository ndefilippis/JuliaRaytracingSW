using GeophysicalFlows, CairoMakie, Printf;
using Random: seed!

dev = CPU()
n = 256
stepper = "AB3"  # timestepper
 nsteps = 20000          # total number of time-steps
 nsubs  = 50             # number of time-steps for plotting (nsteps must be multiple of nsubs)

function compute_parameters(rd, intervortex_radius)
    c₁ = 3.2
    c₂ = 0.36
    U = 1.0;
    H = 0.5;
    ρ1 = 1;
    
    μ = c₂*U/(rd*log(intervortex_radius) - rd*log(c₁*rd)); # bottom drag
    ρ2 = 1 / (1 - 2*rd^2/H)*ρ1
    V = U*intervortex_radius/rd;
    
    return μ, ρ2, V
end

L = 2π                   # domain size
rd = L/20
intervortex_radius = L/5
μ, ρ2, V = compute_parameters(rd, intervortex_radius)            
β = 0                    # the y-gradient of planetary PV

nlayers = 2              # number of layers
f₀, g = 1.0, 1.0            # Coriolis parameter and gravitational constant
H = [0.5, 0.5]           # the rest depths of each layer
ρ = [1.0, ρ2]           # the density of each layer
nν = 4;
ν = (2/n/3)^(2*nν);

U = zeros(nlayers)       # the imposed mean zonal flow in each layer
U[1] =  1.0
U[2] = -1.0

dx = L/n;
dt = 0.5 * dx/V         # timestep
println(@sprintf("bottom drag: %.5f, time step: %.4f, second density: %.4f", μ, dt, ρ2));


prob = MultiLayerQG.Problem(nlayers, dev; nx=n, Lx=L, f₀, g, H, ρ, U, μ, β, nν, ν,
                            dt, stepper, aliased_fraction=1/3)
sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
x, y = grid.x, grid.y

seed!(1234) # reset of the random number generator for reproducibility
q₀  = 1e-2 * device_array(dev)(randn((grid.nx, grid.ny, nlayers)))
#q₀h = prob.timestepper.filter .* rfft(q₀, (1, 2)) # apply rfft  only in dims=1, 2
#q₀  = irfft(q₀h, grid.nx, (1, 2))                 # apply irfft only in dims=1, 2

MultiLayerQG.set_q!(prob, q₀)
E = Diagnostic(MultiLayerQG.energies, prob; nsteps)
diags = [E]

filepath = "."
filename = joinpath(filepath, "2layer.jld2")
if isfile(filename); rm(filename); end

get_sol(prob) = prob.sol # extracts the Fourier-transformed solution

function get_u(prob)
  sol, params, vars, grid = prob.sol, prob.params, prob.vars, prob.grid

  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, params, grid)
  @. vars.uh = -im * grid.l * vars.ψh
  invtransform!(vars.u, vars.uh, params)

  return vars.u
end

out = Output(prob, filename, (:sol, get_sol), (:u, get_u))

Lx, Ly = grid.Lx, grid.Ly

title_KE = Observable(@sprintf("μt = %.2f", μ * clock.t))
q = Observable(Array(vars.q[:, :, 1]))
KE = Observable(Point2f[(μ * E.t[1], E.data[1][1][1])])

Egrid = Observable(Array(0.5 * (vars.u[:,:,1].^2 + vars.v[:,:,1].^2)));
Eh = @lift rfft(Array($Egrid))                         # Fourier transform of energy density
krEhr = @lift FourierFlows.radialspectrum($Eh, grid, refinement = 1)
kr = @lift $krEhr[1]
Ehr = @lift vec(abs.($krEhr[2]))

fig = Figure(size=(1000, 600))

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
            limits = ((-0.1, 2.6), (1e-9, 5)))
axKEspec = Axis(fig[1, 3],
            xlabel = L"k_r",
            ylabel = L"\int |\hat{E}| k_r \mathrm{d}k_\theta",
            xscale = log10,
            yscale = log10,
            title = "Radial energy spectrum",
            limits = ((0.3, 1e2), (1e0, 1e5)))


heatmap!(axq, x, y, q; colormap = :balance)
ke = lines!(axKE, KE; linewidth = 3)
kespec = lines!(axKEspec, kr, Ehr; linewidth = 2)

startwalltime = time()

frames = 0:round(Int, nsteps / nsubs)

record(fig, "debug.mp4", frames, framerate = 18) do j
  if j % (1000 / nsubs) == 0
    cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
    u_max = maximum([maximum(abs.(vars.u)), maximum(abs.(vars.v))])

    log = @sprintf("step: %04d, t: %.1f, cfl: %.4f, KE₁: %.3e, u_max: %.5e, walltime: %.2f min",
                   clock.step, clock.t, cfl, E.data[E.i][1][1], u_max, (time()-startwalltime)/60)

    println(log)
    flush(stdout)
  end
  q[] = vars.q[:, :, 1]
  KE[] = push!(KE[], Point2f(μ * E.t[E.i], E.data[E.i][1][1]))
  title_KE[] = @sprintf("μ t = %.2f", μ * clock.t)

  stepforward!(prob, diags, nsubs)
  MultiLayerQG.updatevars!(prob)
end
