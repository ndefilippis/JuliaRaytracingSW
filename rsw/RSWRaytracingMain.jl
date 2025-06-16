include("IFMAB3.jl")
include("SequencedOutputs.jl")
include("GPURaytracing.jl")
include("RaytracingDriver.jl")

include("RaytracingParameters.jl")
include("RotatingShallowWater.jl")
include("RSWRaytracingDriver.jl")

using .RSWRaytracingDriver
using .RaytracingDriver

function start!()
    start_raytracing!()
end

start!()
