module Parameters
    # Domain parameters
    L  = 2π
    nx = 512

    # Equation parameters

    f = 3.0
    Cg = 1.0

    nν = 4
    νtune  = 20.0

    # Time stepper parameters
    cfltune = 0.01
    use_filter = (νtune == 0)
    filter_order = 8
    aliased_fraction = 1/3

    # Output and timing parameters
    spinup_T = 1200.0
    T = 1300.0
    output_dt = 0.025/f
    diag_dt = 0.5

    max_writes = 100
    base_filename= "rsw"


    # Initial condition parameters
    random_initial_condition = true
    Kg = (10, 13)
    ag = 0.2

	Kw = (0, 5)
    aw = 0.1

    # If no random initial condition, specify a snapshot file to load from
    snapshot_file = "/vast/nad9961/rsw/62221567/rsw.000000.jld2"
    snapshot_key = "snapshots/sol/391376"
end
