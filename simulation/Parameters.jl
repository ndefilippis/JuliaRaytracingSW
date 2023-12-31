# Parameters for two layer flows using GeophysicalFlows.jl

module Parameters
# Integrator parameters
stepper = "FilteredRK4"
dt = 1e-3
nsteps = 10000
nx = 128 # number of grid points

# Domain parameters
L = 2π                   # domain size
r = 5e-2                 # bottom drag
beta = 0                    # the y-gradient of planetary PV

nlayers = 2              # number of layers
f, g = 1, 1             # Coriolis parameter and gravitational constant
H = [0.5, 0.5]           # the rest depths of each layer
rho = [1.0, 2.0]           # the density of each layer
nv = 8
v = 0. # small scale dissipation term

U = zeros(nlayers)       # the imposed mean zonal flow in each layer
U[1] =  1.0
U[2] = -1.0

# Initial condition parameters
q0_amplitude = 1e-2*abs(U[1]) # Height of initial q

# Output parameters
filepath = "."
filename = "2layer_test.jld2"
nsubs = 50;
npacketsubs = 5;

# Wavepackets parameters
packetSpinUpDelay = 10000; # Timesteps until we start advecting wavepackets
sqrtNpackets = 15; # Square root of the number of wavepackets;
Npackets = sqrtNpackets^2;
packetVelocityScale = 1e-2; # Factor to scale background velocity, related to the Froude number
Cg = g*H[1];
alpha = 2.; # How close to f the initial wavepacket frequencies are
k0Amplitude = sqrt(alpha^2 - 1)*f/Cg
packet_dt = 1e-4;

end
