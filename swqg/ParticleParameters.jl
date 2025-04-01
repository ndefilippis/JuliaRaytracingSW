module Parameters
    # Domain parameters
    L  = 2π
    nx = 512

    # Equation parameters

    packet_Cg = 3.0 
    background_Cg = 3.0
    f = 0.1 # Maintain a constant deformation radius

    nν = 4
    νtune  = 1

    # Time stepper parameters
    cfltune = 0.1
    use_filter = (νtune == 0)
    filter_order = 8
    aliased_fraction = 1/3

    # Output and timing parameters
    spinup_T = 500.
    T = 7000.
    output_dt = 10.0
    diag_dt = 0.5

    max_writes = 3000
    base_filename = "qgsw"

    # Initial condition parameters
    Kg = (10, 13)
    ag = 0.5

    # Wavepackets output parameters
    packet_base_filename = "packets"
    use_stationary_background_flow = false
    write_gradients = true
    packet_max_writes = 5000
    packet_output_dt = 0.4;

    # Wavepackets parameters
    sqrtNpackets = 128; # Square root of the number of wavepackets;
    Npackets = sqrtNpackets^2;
    ω0 = 2.0; # How close to f the initial wavepacket frequencies are
    k_cutoff = 1000 # The wavenumber that we reset a off a wavepacket
end
