module Parameters
    # Simulation parameters
    Lx = 6π
	nx = 512
    Ro = 0.2 # Rossby number
    ν  = 5.0e-34 * (Lx/(2π))^16
    nν = 8

    # Time stepping parameters
    dt = 5e-2
    stepper = "ETDRK4"

    # Output parameters
    nsteps = 250000
    nsubs  = 250    # Number of timesteps before outputting
    filename = "ty.jld2"
    
    # Initial condition parameters
    k0w_range = (0, 5.)
	k0g_range = (13., 15.)
    Et = parse(Float64, ARGS[2]) # Initial barotropic energy
    Eg = Et  # Initial baroclinic geostrophic balanced energy
    Ew = parse(Float64, ARGS[3]) # Initial baroclinic inertia-gravity wave energy
end
