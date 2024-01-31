# Parameters for two layer flows using GeophysicalFlows.jl
module Parameters
using Printf

# Integrator parameters
stepper = "FilteredRK4"
nsteps = 200000
nx = 256 # number of grid points

# Domain parameters
L = 2π                   # domain size

f, g = 1., 1.             # Coriolis parameter and gravitational constant
deformation_radius = L/40
intervortex_radius = L/5
avg_U = 0.1
H = [0.5, 0.5]           # the rest depths of each layer
nv = 8
v = 0. # small scale dissipation term
cfl_factor = 0.05

# Initial condition parameters
q0_amplitude = 1e-2*avg_U # Height of initial q

# Output parameters
nsubs = 50;
filepath = "."
output_filename = "2layer_test.jld2"
snapshot_filename = @sprintf("initial_condition_%dx%d_U=%.2f.jld2", nx, nx, avg_U)

end
