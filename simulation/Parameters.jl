# Parameters for two layer flows using GeophysicalFlows.jl

module Parameters
# Integrator parameters
stepper = "FilteredETDRK4"
nsteps = 10000
nx = 128 # number of grid points

# Domain parameters
L = 2π                   # domain size

f, g = 1., 1.             # Coriolis parameter and gravitational constant
deformation_radius = L/60
intervortex_radius = L/5
avg_U = 0.3
H = [0.5, 0.5]           # the rest depths of each layer
nv = 8
v = 0. # small scale dissipation term

# Initial condition parameters
q0_amplitude = 1e-2*avg_U # Height of initial q

# Output parameters
filepath = "."
filename = "2layer_test.jld2"
nsubs = 50;
end
