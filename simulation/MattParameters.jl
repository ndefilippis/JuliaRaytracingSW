# Parameters for two layer flows using GeophysicalFlows.jl
module Parameters
using Printf

# Integrator parameters
nx = 384 # number of grid points

# Domain parameters
Ld = 15e3				  # deformation radius
Lx = 25 * 2π * Ld         # domain size
Lx += 1e-5		# Temporary fix to deal with issue with radialspectrum

f, g = 1e-4, 9.81             # Coriolis parameter and gravitational constant
H0 = 4000.
H = [H0/2, H0/2]           # the rest depths of each layer
nv = 8
v = 0. # small scale dissipation term
cfl_factor = 0.05

b2 = 1
b1 = 4 * f^2 * Ld^2 / H0 + b2

U0 = 0.01
U = [2*U0, 0.]
b = [b1, b2]

kappa_star = 0.1
μ = 2 * U0 / Ld * kappa_star

# Initial condition parameters
q0_amplitude = 1e-3*U0 # Height of initial q

# Integrator parameters
Ti = Ld / U0
tmax = 100 * Ti
dt = 60 * 20.
dtsnap = 60 * 60 * 24 * 5
nsubs = Int(dtsnap / dt)
nsteps = ceil(Int, ceil(Int, tmax / dt) / nsubs) * nsubs
stepper = "FilteredRK4"

# Output parameters
nsubs = 50;
filepath = "."
output_filename = "2layer_test.jld2"
snapshot_filename = @sprintf("initial_condition_%dx%d_matt.jld2", nx, nx)

end
