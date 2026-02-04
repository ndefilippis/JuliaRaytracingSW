module Parameters
    # Domain parameters
    L  = 2π
    nx = 512

    # Equation parameters

    f = 3.0
    Cg = 1.0

    nν = 4
    νtune  = 10.0

    # Time stepper parameters
    cfltune = 0.1
    use_filter = (νtune == 0)
    filter_order = 8
    aliased_fraction = 1/3

    scale_factor = 0.5
    # Output and timing parameters
    spinup_T = scale_factor*800.
    T = scale_factor*2800.
    output_dt = 0.2/f
    diag_dt = 0.5/f

    max_writes = 300
    base_filename= "qgsw"

    # Initial condition parameters
    Kg = (10, 13)
    ag = scale_factor * 0.5
end
