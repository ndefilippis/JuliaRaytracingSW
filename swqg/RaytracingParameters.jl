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
    T = 5000.
    output_dt = 1.0/f
    diag_dt = 0.5

    max_writes = 3000
    base_filename = "qgsw"

    # Initial condition parameters
    Kg = (10, 13)
    ag = 0.3

    # Wavepackets output parameters
    packet_base_filename = "packets"
    packet_max_writes = 5000
    packet_output_dt = 0.2/f;

    # Wavepackets parameters
    sqrtNpackets = 128; # Square root of the number of wavepackets;
    Npackets = sqrtNpackets^2;
    ω0 = 2.f0 * f; # How close to f the initial wavepacket frequencies are
    k_cutoff = 60.f0 # The wavenumber that we reset a off a wavepacket
end