# Parameters for two layer flows using GeophysicalFlows.jl
module Parameters
using Printf
using FourierFlows
using JLD2

# Integrator parameters
stepper = "FilteredAB3"
# nsteps = 100000
total_time = 2000.
nx = 512 # number of grid points
device = GPU()

# Domain parameters
Lx = 6π  # domain size

# Initial condition parameters
bc_energy = 0.1
bt_energy = 0.1
k0_min = 13
k0_max = 15

# Simulation parameters
f = 1.             # Coriolis parameter and gravitational constant
deformation_radius = 1
H0 = 1
H = [H0/2, H0/2]           # the rest depths of each layer
nv = 8
v = 0. # small scale dissipation term

dx = Lx / nx
avg_U = sqrt(bc_energy) + sqrt(bt_energy)
dt = 0.1 * dx / avg_U

b2 = 1.
b1 = 4 * f^2 * deformation_radius^2/H0 + b2
μ = 0

U = [0, 0]
b = [b1, b2]



# Set nsteps
nsteps = Int(ceil(total_time / dt))
# Output parameters
nsubs = round(Int, 1/4/dt); # Save every 6 hours
filepath = "."
output_filename = "2layer_freely_evolving.jld2"
snapshot_filename = @sprintf("initial_condition_%dx%d_U=%.2f_freely_evolve.jld2", nx, nx, avg_U)

end
