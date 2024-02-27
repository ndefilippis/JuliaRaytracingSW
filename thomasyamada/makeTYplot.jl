include("TYUtils.jl")
using CairoMakie
using FourierFlows
using FourierFlows: parsevalsum2
using JLD2
using Printf
using LinearAlgebra: ldiv!
using .TYUtils: decompose_balanced_wave

function get_vars(solution, grid)
    ζt = Array{Float64}(undef, (grid.nx, grid.ny))
    qc = Array{Float64}(undef, (grid.nx, grid.ny))
    ucy = Array{Float64}(undef, (grid.nx, grid.ny))
    vcx = Array{Float64}(undef, (grid.nx, grid.ny))
    pc = Array{Float64}(undef, (grid.nx, grid.ny))
    
    ζth = deepcopy(solution[:,:,1])
    uch = deepcopy(solution[:,:,2])
    vch = deepcopy(solution[:,:,3])
    pch = deepcopy(solution[:,:,4])
    
    ldiv!(ζt, grid.rfftplan, ζth)
    ldiv!(ucy, grid.rfftplan, im * grid.l  .* uch)
    ldiv!(vcx, grid.rfftplan, im * grid.kr .* vch)
    ldiv!(pc, grid.rfftplan, pch)
    
    qc = vcx - ucy - pc
    return (ζt, qc)
end

function create_fig(grid)
    fig = Figure(size=(1200,400))
    axζ = Axis(fig[1,1][1,1]; title="ζ_T")
    axq = Axis(fig[1,2][1,1]; title="q_C")
    axKEspec = Axis(fig[2,1],
            xlabel = L"k_r",
            ylabel = L"Energy",
            xscale = log10,
            yscale = log10,
            title = "Radial energy spectrum",
            aspect = 1,
            limits = ((1.0, maximum(grid.kr)), (1e-9, 1)))
    
    return fig, axζ, axq, axKEspec
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
    jldopen(jldfile, "r") do file
        nx = file["grid/nx"]
        ny = file["grid/ny"]
        Lx = file["grid/Lx"]
        Ly = file["grid/Ly"]
        
        grid = TwoDGrid(CPU(); nx, Lx, ny, Ly, aliased_fraction=0, T=Float64)
        frames = keys(file["snapshots/sol"])
        t = Observable(file["snapshots/t"][frames[1]])
        solution = Observable(file["snapshots/sol"][frames[1]])
        fig, axζ, axq, axKEspec = create_fig(grid)
        
        @lift axKEspec.title = @sprintf("t = %f", $t)
        ζtqc = @lift get_vars($solution, grid)
        ζt = @lift $ζtqc[1]
        qc = @lift $ζtqc[2]
        
        ζcolorrange = @lift (-maximum(abs.($ζt)), maximum(abs.($ζt)))
        qcolorrange = @lift (-maximum(abs.($qc)), maximum(abs.($qc)))
        ζhm = heatmap!(axζ, grid.x, grid.y, ζt; colormap = :balance, colorrange=ζcolorrange)
        qhm = heatmap!(axq, grid.x, grid.y, qc; colormap = :balance, colorrange=qcolorrange)
        Colorbar(fig[1,1][2, 1], ζhm, vertical=false, flipaxis = false)
        Colorbar(fig[1,2][2, 1], qhm, vertical=false, flipaxis = false)
        
        energies = @lift kinetic_energy_spectrum($solution, grid)
        kr = @lift $energies[1][1]
        Etr = @lift vec(abs.($energies[1][2])) .+ 1e-9
        Egr = @lift vec(abs.($energies[2][2])) .+ 1e-9
        Ewr = @lift vec(abs.($energies[3][2])) .+ 1e-9
        
        lines!(axKEspec, kr, Etr, label="E_T"; linewidth = 2)
        lines!(axKEspec, kr, Egr, label="E_G"; linewidth = 2)
        lines!(axKEspec, kr, Ewr, label="E_W"; linewidth = 2)
        axislegend(axKEspec)
        ylims!(axKEspec, 1e-9, 5 * max(maximum(Etr[]), maximum(Egr[]), maximum(Ewr[])))
        println("Creating plot...")
		step = 0
        record(fig, "thomas_yamada_energy.mp4", frames, framerate = 18) do frame
            if(step % 300 == 0)
				println(step)
			end
			solution[] = file["snapshots/sol"][frame]
            t[] = file["snapshots/t"][frame]
        end
        println("Plot thomas_yamada_energy.mp4 created")
    end
end
        
make_plot(ARGS[1])
