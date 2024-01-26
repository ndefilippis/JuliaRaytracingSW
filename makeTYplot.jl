include("TYdriver.jl")
using CarioMakie
using FourierFlows
using FourierFlows: parsevalsum2
using JLD2
using .Driver

function create_fig()
    fig = Figure(size=(1200,400))
    axζ = Axis(fig[1,1][1,1]; title="ζ_T")
    axq = Axis(fig[1,2][1,1]; title="q_C")
    axE = Axis(fig[1,3][1,1], xlabel="t", ylabel="bc E", limits=((0, Parameters.nsteps * prob.clock.dt), (0, 10.)))

    ζhm = heatmap!(axζ, prob.grid.x, prob.grid.y, ζt; colormap = :balance)
    qhm = heatmap!(axq, prob.grid.x, prob.grid.y, qc; colormap = :balance)

    Colorbar(fig[1,1][2, 1], ζhm, vertical=false, flipaxis = false)
    Colorbar(fig[1,2][2, 1], qhm, vertical=false, flipaxis = false)
    bc = lines!(axE, baroclinic, label="bc E"; linewidth = 3)
    bt = lines!(axE, barotropic, label="bt E"; linewidth = 3)
    axislegend(axE) 
    
    return fig
end

function kinetic_energy_spectrum(solution, grid)
    Gh, Wh = decompose_balanced_wave(solution, grid)
    KEth = solution[:,:,1].^2
    KEgh = Gh[:,:,1].^2 + Gh[:,:,2].^2
    KEwh = Wh[:,:,1].^2 + Wh[:,:,2].^2
    KEtr = FourierFlows.radialspectrum(KEth, grid, refinement = 1)  
    KEgr = FourierFlows.radialspectrum(KEgh, grid, refinement = 1)
    KEwr = FourierFlows.radialspectrum(KEwh, grid, refinement = 1)
    return (KEtr, KEgr, KEwr)
end

function make_plot(jldfile)
    grid = T
    
    jldopen(jldfile, "r") do file
        frames = keys(file["snapshots/sol"])
        solution = Observable(file["snapshots/sol"])
        energies = @lift kinetic_energy_spectrum($solution, grid)
        kr = @lift $energies[1][1]
        Etr = @lift vec(abs.($energies[1][2])) .+ 1e-9
        Egr = @lift vec(abs.($energies[2][2])) .+ 1e-9
        Ewr = @lift vec(abs.($energies[3][2])) .+ 1e-9
        record(fig, "thomas_yamada_energy.mp4", frames, framerate = 18) do j
            solution[] = sol
            ζt[] = vars.ζt
            qc[] = vars.qc
            baroclinic[] = push!(baroclinic[], Point2f(bcE.t[bcE.i], bcE.data[bcE.i][1]))
            barotropic[] = push!(barotropic[], Point2f(btE.t[btE.i], btE.data[btE.i][1]))
            
            stepforward!(prob, diags, nsubs)
            updatevars!(prob)
            saveoutput(out)
        end
    end
end