module Parameters
    # Domain parameters
    L  = 2π
    nx = 2048

    # Equation parameters

    f = 3.0
    Cg = 1.0

    nν = 4
    νtune  = 1.0

    # Time stepper parameters
    cfltune = 0.1
    use_filter = (νtune == 0)
    filter_order = 8
    aliased_fraction = 1/3

    # Output and timing parameters
    spinup_T = 1000.
    T = 3100.
    output_dt = 1.0
    diag_dt = 0.1

    max_writes = 300
    base_filename= "rsw"


    # Initial condition parameters
    Kg = (10, 13)
    ag = 0.5

    Kw = (0, 5)
    aw = 0.1
end
