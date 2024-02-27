module Parameters
    # Simulation parameters
    nx = 512
    Ro = 0.2 # Rossby number
    ν  = 2.4e-34
    nν = 8

    # Time stepping parameters
    dt = 1e-2
    stepper = "ETDRK4"

    # Output parameters
    nsteps = 500000
    nsubs  = 250    # Number of timesteps before outputting
    filename = "ty.jld2"
    
    # Initial condition parameters
    k0w = 15
	k0g = 6
    Et = 0.5 # Initial barotropic energy
    Eg = Et  # Initial baroclinic geostrophic balanced energy
    Ew = 0 # Initial baroclinic inertia-gravity wave energy
end
