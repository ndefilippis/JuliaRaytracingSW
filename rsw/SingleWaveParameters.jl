module Parameters

function compute_packet_positions(Lx, Npackets)
    
end

# Domain parameters
L  = 2π
nx = 1024

# Equation parameters

background_Cg = 1.0
packet_Cg = background_Cg
Cg = packet_Cg
f = 3.0 * Cg # Maintain a constant deformation radius

nν = 4
νtune  = 10

# Time stepper parameters
cfltune = 0.1
use_filter = (νtune == 0)
filter_order = 8
aliased_fraction = 1/3

# Output and timing parameters
packet_spinup_T = 1000.
spinup_T = 1000.
T = 1200.
output_dt = 0.1
diag_dt = 0.5

max_writes = 300
base_filename = "rsw"

# Initial condition parameters
Kg = (10, 13)
ag = 0.6

Kw = (0, 1)
aw = 0.0
aw_2 = 0.02

# Initial wave parameters
x0 = 0
y0 = 0
k0_idx = 40
l0_idx = 40
sgn = 1
phase = 0
env_size = L/80

# Wavepackets output parameters
packet_base_filename = "packets"
use_stationary_background_flow = false
write_gradients = true
packet_max_writes = 300
packet_output_dt = 0.1

k_cutoff = 100.0 * f / Cg # The wavenumber that we reset a off a wavepacket
end
