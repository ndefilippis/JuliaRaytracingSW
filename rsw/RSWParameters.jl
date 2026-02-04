module Parameters
    # Domain parameters
    L  = 2π
    nx = 512

    # Equation parameters

    f = 3.0
    Cg = 1.0

    nν = 4
    νtune  = 0.05

    # Time stepper parameters
    cfltune = 0.01
    use_filter = (νtune == 0)
    filter_order = 8
    aliased_fraction = 1/3

    # Output and timing parameters
    spinup_T = 2000.0
    T = 2050.0
    output_dt = 1.0
    diag_dt = 0.5

    max_writes = 100
    base_filename= "rsw"


    # Initial condition parameters
    random_initial_condition = true
    Kg = (6, 10)
    ag = 0.05

	Kw = (0, 3)
    aw = 0.1

    # If no random initial condition, specify a snapshot file to load from
    snapshot_file = "/vast/nad9961/rsw/62221567/rsw.000000.jld2"
    snapshot_key = "snapshots/sol/391376"
end
