# Parameters for two layer raytracing simulation

module Parameters

# Integrator parameters
stepper = "FilteredETDRK4"
nsteps = 100
#nx = 512 # number of grid points

# Domain parameters
L = 2π                   # domain size

# Output parameters
filepath = "."
filename = "packets.jld2"
nsubs = 50;
npacketsubs = 5;

initial_condition_file = "initial_condition.jld2"

# Wavepackets parameters
packetSpinUpDelay = 0; # Timesteps until we start advecting wavepackets
sqrtNpackets = 2; # Square root of the number of wavepackets;
Npackets = sqrtNpackets^2;
initialFroudeNumber = 1e-1; # Scale initial steady background field to achieve this Froude number
corFactor = 2.; # How close to f the initial wavepacket frequencies are
packetStepsPerBackgroundStep = 10
end
