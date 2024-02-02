# Parameters for two layer flows using GeophysicalFlows.jl
module Parameters
using Printf

function compute_parameters(rd, l, avg_U, H, f)
    c₁ = 3.2
    c₂ = 0.36
    l_star = l/rd
    b2 = 1.;
    
    U = avg_U / l_star
    μ = 2*c₂*U/(rd*log(l_star/c₁)); # bottom drag
    b1 = 4 * f^2 * rd^2/H + b2
    
    # U = avg_U/l_star/sqrt(log(l_star));
    # V = U * l_star * log(l_star);
    
    return μ, b1, U
end

# Integrator parameters
stepper = "FilteredRK4"
nsteps = 200000
nx = 1024 # number of grid points

# Domain parameters
Lx = 2π                   # domain size

f = 1.             # Coriolis parameter and gravitational constant
deformation_radius = 1/25
intervortex_radius = 1/5
avg_U = 0.1
H = [0.5, 0.5]           # the rest depths of each layer
nv = 8
v = 0. # small scale dissipation term

dx = Lx / nx
dt = 0.05 * dx / avg_U

b2 = 1.
μ, b1, shear_strength = compute_parameters(deformation_radius, intervortex_radius, avg_U, sum(H), f)

U = [shear_strength, -shear_strength]
b = [b1, b2]

# Initial condition parameters
q0_amplitude = 1e-2*avg_U # Height of initial q

# Output parameters
nsubs = 50;
filepath = "."
output_filename = "2layer_test.jld2"
snapshot_filename = @sprintf("initial_condition_%dx%d_U=%.2f.jld2", nx, nx, avg_U)

end
