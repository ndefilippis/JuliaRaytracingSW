# Parameters for two layer raytracing simulation

module Parameters
using FourierFlows

# Integrator parameters
stepper = "FilteredAB3"
total_time = 2000.0f0

# Device
device = GPU()

# Domain parameters
L = 2π                   # domain size

# Output parameters
filepath = "."
filename = "packets.jld2"
nsubs = 1;
npacketsubs = 10;
max_writes = 1000;

#initial_condition_file = "initial_condition.jld2"
initial_condition_file = "initial_conditions/initial_condition_512x512_U=1.10_freely_evolve.jld2"

# Wavepackets parameters
packetSpinUpDelay = 0; # Timesteps until we start advecting wavepackets
sqrtNpackets = 20; # Square root of the number of wavepackets;
Npackets = sqrtNpackets^2;
initialFroudeNumber = 1; # Scale initial steady background field to achieve this Froude number
corFactor = 2.f0; # How close to f the initial wavepacket frequencies are
k_cutoff = 100.f0
packetStepsPerBackgroundStep = 1
end