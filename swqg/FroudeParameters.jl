module Parameters
    # Domain parameters
    L  = 2π
    nx = 512

    # Equation parameters

    background_Cg = parse(Float32, ARGS[3])
    packet_Cg = background_Cg
    Cg = packet_Cg
    f = 3.0 * Cg # Maintain a constant deformation radius

    nν = 4
    νtune  = 1

    # Time stepper parameters
    cfltune = 0.1
    use_filter = (νtune == 0)
    filter_order = 8
    aliased_fraction = 1/3

    # Output and timing parameters

    T = parse(Float32, ARGS[4])
    spinup_T = T/10.0

    output_dt = 10.0/f
    diag_dt = 0.5/f

    max_writes = 500
    base_filename = "swqg"

    # Initial condition parameters
    Kg = (10, 13)
    ag = parse(Float32, ARGS[1])

    # Wavepackets output parameters
    packet_base_filename = "packets"
    use_stationary_background_flow = false
    write_gradients = true
    packet_max_writes = 3000
    packet_output_dt = 1.0;

    # Wavepackets parameters
    sqrtNpackets = 256; # Square root of the number of wavepackets;
    Npackets = sqrtNpackets^2;
    ω0 = sqrt(f^2 + Cg^2 * parse(Float32, ARGS[2])^2); # How close to f the initial wavepacket frequencies are
    k_cutoff = 1000*f/Cg # The wavenumber that we reset a off a wavepacket
end
