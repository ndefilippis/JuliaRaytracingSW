module Parameters
    # Domain parameters
    L  = 2π
    nx = 512

    # Equation parameters
    
    Cg = parse(Float32, ARGS[2])
    f = 3.0 * Cg

    nν = 4
    νtune  = 1

    # Time stepper parameters
    cfltune = 0.1
    use_filter = (νtune == 0)
    filter_order = 8
    aliased_fraction = 1/3

    # Output and timing parameters
    spinup_T = 2500.
    T = 2600.
    output_dt = 0.03/f
    diag_dt = 0.01

    max_writes = 100
    base_filename= "rsw"


    # Initial condition parameters
    Kg = (10, 13)
    ag = 0.3

    Kw = (0, 5)
    aw = parse(Float32, ARGS[1])
end
