module Parameters
    # Simulation parameters
    nx = 128
    Ro = 0.2 # Rossby number
    ν  = 3.5e-25
    nν = 8

    # Time stepping parameters
    device = "GPU"
    dt = 5e-2
    stepper = "ETDRK4"

    # Output parameters
    nsteps = 30000
    nsubs  = 50    # Number of timesteps before outputting
    filename = "ty.jld2"
    
    # Initial condition parameters
    k0 = 6
    Et = 0.3 # Initial barotropic energy
    Eg = Et  # Initial baroclinic geostrophic balanced energy
    Ew = 0.3 # Initial baroclinic inertia-gravity wave energy
end