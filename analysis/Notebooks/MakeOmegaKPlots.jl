using CairoMakie
using FourierFlows
using FourierFlows: parsevalsum2
using JLD2
using Printf
using LinearAlgebra: ldiv!
using AbstractFFTs
include("../../utils/ExactRadialSpectrum.jl")

function frequency_filters(N)
    # Returns negative zero, and positive frequency filters
    if (N % 2 == 0)
        N_half = Int(N//2)+1
        return (N_half-1):-1:2, N_half, (N_half+1):N
    else
        N_half = Int((N+1)//2)
        return (N_half-1):-1:1, N_half, (N_half+1):N
    end
    
end

function read_frequency_file_to_radial_data(radii, k_max, weight_matrix, fft_directory, output_file)  
    initial_data_file = jldopen(@sprintf("%s/1/radial_data_k=001.jld2", fft_directory))
    N = length(initial_data_file["t"])
    t = initial_data_file["t"]
    dt = t[2] - t[1]
    ω = fftshift(fftfreq(N, 1/dt)) * 2*pi

    C0_data = zeros(N, length(radii))
    Cp_data = zeros(N, length(radii))
    Cn_data = zeros(N, length(radii))
    
    for k_idx=1:k_max
        data_file = nothing
        filename = @sprintf("%s/%d/radial_data_k=%03d.jld2", fft_directory, ceil(Int,k_idx/4), k_idx)
        
        if !isfile(filename)
            println("Missing file: " * filename)
            continue
        end
        println(filename)
        data_file = jldopen(filename)
        
        chosen_weights = [weights[k_idx, :] for weights=weight_matrix]
        #C0slice = eachslice(abs2.(data_file["c0t"]), dims=1)
        #Cpslice = eachslice(abs2.(data_file["c+t"]), dims=1)
        #Cnslice = eachslice(abs2.(data_file["c-t"]), dims=1)
        C0slice = eachslice(abs2.(data_file["ugt"]) + abs2.(data_file["vgt"]) + abs2.(data_file["ηgt"]), dims=1)
        Cpslice = eachslice(abs2.(data_file["uwt"]) + abs2.(data_file["vwt"]) + abs2.(data_file["ηwt"]), dims=1)
        Cnslice = eachslice(abs2.(data_file["uwt"]) + abs2.(data_file["vwt"]) + abs2.(data_file["ηwt"]), dims=1)
        C0_data += chosen_weights' .* C0slice
        Cp_data += 0.5*chosen_weights' .* Cpslice
        Cn_data += 0.5*chosen_weights' .* Cnslice
    
        close(data_file)
        data_file = nothing
        GC.gc()
    end

    C0_data = fftshift(C0_data, 1)
    Cp_data = fftshift(Cp_data, 1)
    Cn_data = fftshift(Cn_data, 1)
    output = jldopen(output_file, "w")
    output["C0"] = C0_data
    output["Cp"] = Cp_data
    output["Cn"] = Cn_data
    output["K"] = radii
    output["ω"] = ω
    close(output)

    return N, ω, C0_data, Cp_data, Cn_data
end

function create_total_plot(N, sqrtgH, f0, C0_data, Cp_data, Cn_data, grid, radii, ω, ax, label, color_idx)
    N_neg, N_half, N_pos = frequency_filters(N)
    K_max = grid.nkr
    K_d = f0/sqrtgH
    
    C0_ndata = (C0_data) * grid.dx / grid.nx
    Cp_ndata = (Cp_data) * grid.dx / grid.nx
    Cn_ndata = (Cn_data) * grid.dx / grid.nx
    T_ndata  = C0_ndata + Cp_ndata + Cn_ndata
    y_data = ω[N_pos]/f0
    data = T_ndata[N_neg, :] + T_ndata[N_pos, :]
    summed_data = sum(data, dims=2)[:]
    lines!(ax, y_data, summed_data, linewidth=3, label=label, colormap=:berlin10, color=color_idx, colorrange=(1,10))
end

function create_wavenumber_frequency_plot(N, sqrtgH, f0, C0_data, Cp_data, Cn_data, grid, radii, ω)
    N_neg, N_half, N_pos = frequency_filters(N)
    K_max = grid.nkr
    K_d = f0/sqrtgH
    
    C0_ndata = (C0_data) * grid.dx / grid.nx
    Cp_ndata = (Cp_data) * grid.dx / grid.nx
    Cn_ndata = (Cn_data) * grid.dx / grid.nx
    W_data = Cp_ndata + Cn_ndata
    T_ndata  = C0_ndata + Cp_ndata + Cn_ndata
    data_order = [C0_ndata, W_data, T_ndata]
    
    fig_ωk = Figure(size=(3300, 1800), fontsize=100)
    fig_total_wave = Figure(size=(800, 800), fontsize=100)
    
    Label(fig_ωk[-1, 1:3], "RSW Power Spectrum")
    Label(fig_ωk[0, 1], "Geostrophic flow")
    #Label(fig_ωk[0, 2], "Positive waves")
    #Label(fig_ωk[0, 3], "Negative waves")
    Label(fig_ωk[0, 2], "Waves")
    Label(fig_ωk[0, 3], "Total")
    axis_ωk_options = (xscale=log10, yscale=log10,
        xticks = [1, 10], yticks=[1, 10, 100],
        xminorticks=IntervalsBetween(10), yminorticks=IntervalsBetween(10),
        limits=(((radii[1])/K_d, radii[end-1]/K_d), (ω[N_half+1]/f0, ω[end]/f0)),
        xminorticksvisible = true, yminorticksvisible = true)

    fig_sum = Figure(size=(2500, 1800), fontsize=100)
    Label(fig_sum[-1, 1:3], "RSW Power Spectrum")
    Label(fig_sum[0, 1], "Geostrophic")
    #Label(fig_sum[0, 2], "Positive waves")
    #Label(fig_sum[0, 3], "Negative waves")
    Label(fig_sum[0, 2], "All waves")
    Label(fig_sum[0, 3], "Total")
    axis_sum_options = (xscale=log10, yscale=log10,
       limits=((ω[N_half+1]/f0, 4*ω[end]/f0), (1e2, 1e10)),
        xticks = [0.1, 1, 10, 100])
    axis_sum = nothing
    axis_ωk = nothing
    for data_type = 1:3
        y_data = ω[N_pos]/f0
        ω10_idx = findfirst(>=(10), y_data)
        x_data = (radii[1:end])/K_d
        ylabel_ωk="ω/f"
        ylabel_sum="E(ω)"
        test_data = log.(data_order[data_type])
        min_val = minimum(test_data)
        max_val = maximum(test_data)
        colorlimits=(-10, 20)
        levels=range(colorlimits[1], colorlimits[2], length=10)
        for freq_sign = 1:1
            data = data_order[data_type]
            if freq_sign == 1
                data = data[N_pos, :]
            elseif freq_sign == 2
                data = data[N_neg, :]
                ylabel_ωk="-ω/f"
                ylabel_sum="E(-ω)"
            else
                data = data[N_neg, :] + data[N_pos, :]
                ylabel_ωk = "|ω|/f"
                ylabel_sum = "E(|ω|)"
            end
            axis_ωk = Axis(fig_ωk[freq_sign, data_type]; axis_ωk_options...)
            axis_sum = Axis(fig_sum[freq_sign, data_type]; axis_sum_options...)
            if(data_type == 1)
                axis_ωk.ylabel = ylabel_ωk
                axis_sum.ylabel = ylabel_sum
            end
            #if(freq_sign == 3)
                axis_ωk.xlabel = rich("K/K", subscript("d"))
                axis_sum.xlabel = "ω/f"
            #end
            
            cf = contourf!(axis_ωk, x_data, y_data, log.(data'), colormap=:haline, levels=levels, extendlow = :auto, extendhigh = :auto)
            summed_data = sum(data, dims=2)[:]
            lines!(axis_sum, y_data, summed_data, linewidth=3)

            coefficient2 = summed_data[ω10_idx]/(y_data[ω10_idx])^(-2.00)
            coefficient3 = summed_data[ω10_idx]/(y_data[ω10_idx])^(-3.00)
            lines!(axis_sum, y_data, coefficient2*y_data.^(-2.00), linestyle=:dash, color="black", label=rich("ω", superscript("-2")), linewidth=5)
            lines!(axis_sum, y_data, coefficient3*y_data.^(-3.00), linestyle=:dash, color="gray", label=rich("ω", superscript("-3")), linewidth=5)


            #if(data_type == 4 && freq_sign == 3)
            #    axis_total_wave = Axis(fig_total_wave[1, 1]; xlabel="|ω|/f", ylabel="E(|ω|)", axis_sum_options...)
            #    lines!(axis_total_wave, y_data, summed_data, linewidth=3)
            #    lines!(axis_total_wave, y_data, coefficient2*y_data.^(-2.00), linestyle=:dash, color="lightgray", label=rich("ω", superscript("-2")), linewidth=3)
            #    lines!(axis_total_wave, y_data, coefficient3*y_data.^(-3.00), linestyle=:dash, color="black", label=rich("ω", superscript("-3")), linewidth=3)
            #end

            if(freq_sign == 1)
                cb = Colorbar(fig_ωk[2, data_type], cf, vertical = false)
            end
            lines!(axis_ωk, x_data, sqrt.(f0^2 .+ (sqrtgH * K_d*(x_data)).^2)/f0, linestyle=:dash, color="black", alpha=0.75, linewidth=5, label="ω(K)")
        end
    end
    Legend(fig_sum[2,1:3], axis_sum, orientation=:horizontal, patchsize = (80, 20))
    Legend(fig_ωk[3,1:3], axis_ωk, orientation=:horizontal, patchsize = (80, 20))
    
    colsize!(fig_ωk.layout, 1, Aspect(1, 1.0))
    colsize!(fig_ωk.layout, 2, Aspect(1, 1.0))
    colsize!(fig_ωk.layout, 3, Aspect(1, 1.0))
    #colsize!(fig_ωk.layout, 4, Aspect(1, 1.0))
    #colsize!(fig_ωk.layout, 5, Aspect(1, 1.0))
    resize_to_layout!(fig_ωk)

    colsize!(fig_sum.layout, 1, Aspect(1, 1.0))
    colsize!(fig_sum.layout, 2, Aspect(1, 1.0))
    colsize!(fig_sum.layout, 3, Aspect(1, 1.0))
    #colsize!(fig_sum.layout, 4, Aspect(1, 1.0))
    #colsize!(fig_sum.layout, 5, Aspect(1, 1.0))
    resize_to_layout!(fig_sum)
    
    return fig_ωk, fig_sum, fig_total_wave
end

function start!()
    grid = TwoDGrid(; Lx=2π, nx=512)
    radii, weight_matrix = create_radialspectrum_weights(grid, 3);

    test_input_file = @sprintf("/scratch/nad9961/rsw_fourier/57410496/%d/all_radial_data.jld2", 5)
    test_input = jldopen(test_input_file)
    ω = test_input["ω"]
    N = length(ω)
    close(test_input)
    N_neg, N_half, N_pos = frequency_filters(N)
    
    total_fig = Figure(size=(1500, 1000))
    total_sum_axis = Axis(total_fig[1,1]; xscale=log10, yscale=log10,
        limits=((ω[N_half+1]/2, ω[end]), (1e4, 1e10)),
        xticks = [1e-1, 1e0, 1e1, 1e2])
    for run_idx=5:12
        input_file = @sprintf("/scratch/nad9961/rsw_fourier/57410496/%d/all_radial_data.jld2", run_idx)
        sqrtgH = 0.35 + run_idx * 0.05
        f0 = 3.0 * sqrtgH
        input_file = @sprintf("/scratch/nad9961/rsw_fourier/57410496/%d/all_radial_data.jld2", run_idx)
        input = jldopen(input_file)
        C0_data = input["C0"]
        Cp_data = input["Cp"] 
        Cn_data = input["Cn"]
        ω = input["ω"]
        N = length(ω)
        close(input)
        create_total_plot(N, sqrtgH, f0, C0_data, Cp_data, Cn_data, grid, radii, ω, total_sum_axis, @sprintf("sqrt(gH) = %0.2f", sqrtgH), (run_idx-4))
    end
    total_fig[1,2] = Legend(total_fig, total_sum_axis)
    resize_to_layout!(total_fig)
    save(@sprintf("images/total_spectrum_run=%s.png", "57410496"), total_fig)
end

function read_from_radial_data_file(radial_data_directory)
    input = jldopen(radial_data_directory)
    C0_data = input["C0"]
    Cp_data = input["Cp"] 
    Cn_data = input["Cn"]
    ω = input["ω"]
    N = length(ω)
    close(input)
    return N, ω, C0_data, Cp_data, Cn_data
end

function make_images!(fft_directory, key_name, grid_size, f0, sqrtgH)
    grid = TwoDGrid(; Lx=2π, nx=grid_size)
    radii, weight_matrix = create_radialspectrum_weights(grid, 3);
    output_file = @sprintf("%s/all_radial_data.jld2", fft_directory)
    N, ω, C0_data, Cp_data, Cn_data = read_frequency_file_to_radial_data(radii, grid.nkr-1, weight_matrix, fft_directory, output_file)
    #N, ω, C0_data, Cp_data, Cn_data = read_from_radial_data_file(@sprintf("%s/all_radial_data.jld2", fft_directory))
    
    fig1, fig2, fig3 = create_wavenumber_frequency_plot(N, sqrtgH, f0, C0_data, Cp_data, Cn_data, grid, radii, ω)

    save(@sprintf("images/KOmega_spectrum_run=%s.png", key_name), fig1)
    save(@sprintf("images/Omega_spectrum_run=%s.png", key_name), fig2)
    save(@sprintf("images/total_wave_spectrum_run=%s.png", key_name), fig3)
end
