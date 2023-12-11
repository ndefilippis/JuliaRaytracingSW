# Parameters for two layer flows using GeophysicalFlows.jl

module Parameters

function compute_parameters(rd, intervortex_radius, U, H, ρ1)
    c₁ = 3.2
    c₂ = 0.36
    
    μ = c₂*U/(rd*log(intervortex_radius) - rd*log(c₁*rd)); # bottom drag
    ρ2 = 1 / (1 - 2*rd^2/H)*ρ1
    V = U*intervortex_radius/rd;
    
    return μ, ρ2, V
end

# Integrator parameters
stepper = "FilteredETDRK4"
nsteps = 101#100000
#nx = 512 # number of grid points

# Domain parameters
L = 2π                   # domain size
#rd = 1/20
#intervortex_radius = 1/5

#beta = 0                    # the y-gradient of planetary PV

#nlayers = 2              # number of layers
#f, g = 1, 1             # Coriolis parameter and gravitational constant
#U = zeros(nlayers)       # the imposed mean zonal flow in each layer
#U[1] =  1.0
#U[2] = -1.0
#H = [0.5, 0.5]           # the rest depths of each layer

#rho = [1.0, undef]           # the density of each layer
#μ, ρ[2], V = compute_parameters(rd, intervortex_radius, U[1], H[1], 1.0)
#nv = 8
#v = 0. # small scale dissipation term

#dt = 0.05 * L/V/n

# Initial condition parameters
#q0_amplitude = 1e-2*abs(U[1]) # Height of initial q

# Output parameters
filepath = "."
filename = "2layer_test.jld2"
nsubs = 50;
npacketsubs = 5;

initial_condition_file = "JuliaRaytracing/raytracing/initial_condition.jld2"

# Wavepackets parameters
packetSpinUpDelay = 0; # Timesteps until we start advecting wavepackets
sqrtNpackets = 1#15; # Square root of the number of wavepackets;
Npackets = sqrtNpackets^2;
initialFroudeNumber = 1e-1; # Scale initial steady background field to achieve this Froude number
# Cg = g*H[1];
corFactor = 2.; # How close to f the initial wavepacket frequencies are
#k0Amplitude = sqrt(alpha^2 - 1)*f/Cg
packetStepsPerBackgroundStep = 10
end
