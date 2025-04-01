module Parameters
    # Domain parameters
    L  = 2π
    nx = 512

    # Equation parameters

    f = 3.0
    Cg = 1.0

    nν = 4
    νtune  = 1

    # Time stepper parameters
    cfltune = 0.1
    use_filter = (νtune == 0)
    filter_order = 8
    aliased_fraction = 1/3

    # Output and timing parameters
    spinup_T = 2000.
    T = 2100.
    output_dt = 0.03/f
    diag_dt = 0.05

    max_writes = 3000
    base_filename= "qgsw"

    # Initial condition parameters
    Kg = (10, 13)
    ag = 0.3
end
