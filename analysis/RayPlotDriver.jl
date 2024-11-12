include("MakeRaytracingPlots.jl")
using Printf

@Threads.threads for i=1:15
    create_plots(
        string("/scratch/nad9961/swqg/52585523/", i),
        @sprintf("case_%02d_4096_packets", i),
        30, 1:4:(128^2), 3 * i/10, i/10)
    flush(stdout)
end
