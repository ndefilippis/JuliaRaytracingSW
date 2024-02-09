# Parameters for two layer raytracing simulation

module Parameters
using FourierFlows

# Integrator parameters
stepper = "FilteredRK4"
nsteps = 50000

# Device
dev = GPU()

# Domain parameters
L = 2π                   # domain size

# Output parameters
filepath = "."
filename = "packets2.jld2"
nsubs = 1;
npacketsubs = 1;

initial_condition_file = "initial_condition.jld2"

# Wavepackets parameters
packetSpinUpDelay = 0; # Timesteps until we start advecting wavepackets
sqrtNpackets = 4; # Square root of the number of wavepackets;
Npackets = sqrtNpackets^2;
initialFroudeNumber = 2e-1; # Scale initial steady background field to achieve this Froude number
corFactor = 2.; # How close to f the initial wavepacket frequencies are
packetStepsPerBackgroundStep = 10
end
