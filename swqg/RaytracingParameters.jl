module Parameters
    # Domain parameters
    L  = 2π
    nx = 512

    # Equation parameters

    f = 1e-4
    Cg = 1.0
    background_Cg = Cg
    packet_Cg = Cg

    nν = 4
    νtune  = 10.0
    

    # Time stepper parameters
    cfltune = 0.1
    use_filter = (νtune == 0)
    filter_order = 8
    aliased_fraction = 1/3

    # Output and timing parameters
    spinup_T = 800.
    T = 100_000.
    output_dt = 1000.0
    diag_dt = 500.0

    max_writes = 3000
    base_filename = "qgsw"

    # Initial condition parameters
    use_snapshot_file = false
    snapshot_file = ""
    snapshot_key = ""
    Kg = (10, 13)
    ag = 0.1

    # Wavepackets output parameters
    packet_base_filename = "packets"
    use_stationary_background_flow = true
    write_gradients = true
    packet_max_writes = 500
    packet_output_dt = 100.0;

    # Wavepackets parameters
    sqrtNpackets = 512; # Square root of the number of wavepackets;
    Npackets = sqrtNpackets^2;
    ω0 = 2.f0; # How close to f the initial wavepacket frequencies are
    k_cutoff = 100.0 / Cg # The wavenumber that we reset a off a wavepacket
end
