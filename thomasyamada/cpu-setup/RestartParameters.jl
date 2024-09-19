module Parameters
    # Simulation parameters
    Lx = 6π
    nx = 512
    Ro = 1.0 # Rossby number
    ν  = 5.0e-34 * (Lx/(2π))^16
    nν = 8

    # Time stepping parameters
    startup_dt = 3e-3
    dt = 5e-3
    stepper = "ETDRK4"

    # Output parameters

    startup_nsubs = 1
    startup_nsteps = 1

    restart_file = "/vast/nad9961/thomasyamada_simulation/50225982/startup.jld2"
    restart_frame = "snapshots/sol/500100"
    
    nsteps = 25133 * 2
    nsubs  = 1    # Number of timesteps before outputting
    
    max_writes = 100
    filename = "ty.jld2"
    
    # Initial condition parameters
    k0w_range = (0, 5/3.)
	k0g_range = (10/3., 13/3.)
    at = 0.0 # Initial barotropic energy
    ag = 0.3  # Initial baroclinic geostrophic balanced energy
    aw = 0.0 # Initial baroclinic inertia-gravity wave energy
end
