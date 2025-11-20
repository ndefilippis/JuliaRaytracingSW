module Parameters
    # Domain parameters
    L  = 2π
    nx = 1024

    # Equation parameters
    
    Cg = 1.0
    f = 3.0 * Cg

    nν = 4
    νtune  = 50.0

    # Time stepper parameters
    cfltune = 0.02
    use_filter = (νtune == 0)
    filter_order = 8
    aliased_fraction = 1/3

    # Output and timing parameters
    spinup_T = 300.
    T = 400.
    output_dt = 0.01/f
    diag_dt = 0.1

    max_writes = 100
    base_filename= "rsw"


    # Initial condition parameters
    random_initial_condition = false
    Kg = (10, 13)
    ag = 0.0

    Kw = (0, 1)
    aw = 0.01

    front_initial_condition = true
    Nwaves = 120

    # If no random initial condition, specify a startup file and snapshot to load
    snapshot_file = "/vast/nad996/rsw/62221567/rsw.000000.jld2"
    snapshot_key = "snapshots/sol/391376"
end
