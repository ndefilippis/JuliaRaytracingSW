module Parameters

__revise_mode__ = :eval

# Domain parameters
L  = 2π
nx = 512
# Equation parameters

background_Cg = 1.0
packet_Cg = background_Cg
Cg = packet_Cg
f = 3.0 * Cg # Maintain a constant deformation radius
deformation_radius = 1/6
intervortex_radius = 1

nν = 4
νtune = 100

# Time stepper parameters
cfltune = 0.05
use_filter = (νtune == 0)
filter_order = 8
aliased_fraction = 1/3

# Output and timing parameters
spinup_T = 10000.
T = 20000.
output_dt = 3.0
diag_dt = 0.5

max_writes = 300
base_filename = "2Lqg"

# Initial condition parameters
ug = 0.05
end
