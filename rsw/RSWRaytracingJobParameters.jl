module Parameters
    # Domain parameters
    L  = 2π
    nx = 512

    # Equation parameters

    background_Cg = 1.0
    packet_Cg = background_Cg
    Cg = packet_Cg
    f = 3.0 * Cg # Maintain a constant deformation radius

    nν = 4
    νtune  = 10.0

    # Time stepper parameters
    cfltune = 0.04
    use_filter = (νtune == 0)
    filter_order = 8
    aliased_fraction = 1/3

    # Output and timing parameters
    packet_spinup_T = 400.
    spinup_T = 1000.
    T = 10000.
    output_dt = 10.0/f
    diag_dt = 0.5

    max_writes = 300
    base_filename = "rsw"

    # Initial condition parameters
    Kg = (10, 13)
    ag = parse(Float32, ARGS[1])

    Kw = (0, 5)
    aw = 0.1

    # Wavepackets output parameters
    packet_base_filename = "packets"
    use_stationary_background_flow = false
    write_gradients = true
    packet_max_writes = 300
    packet_output_dt = 1.0

    # Wavepackets parameters
    sqrtNpackets = 128; # Square root of the number of wavepackets;
    Npackets = sqrtNpackets^2;
    ω0 = 2.f0 * f; # How close to f the initial wavepacket frequencies are
    k_cutoff = 100.0 * f / Cg # The wavenumber that we reset a off a wavepacket
end
