module Parameters
    # Simulation parameters
    nx = 384
    Ro = 0.3 # Rossby number
    ν  = 2.4e-34
    nν = 8

    # Time stepping parameters
    dt = 5e-2
    stepper = "ETDRK4"

    # Output parameters
    nsteps = 100000
    nsubs  = 50    # Number of timesteps before outputting
    filename = "ty.jld2"
    
    # Initial condition parameters
    Et = 0.3 # Initial barotropic energy
    Eg = Et  # Initial baroclinic geostrophic balanced energy
    Ew = 0.3 # Initial baroclinic inertia-gravity wave energy
end
