using FourierFlows
using Printf
using CairoMakie
include("RSWTransferFunction.jl")
include("RSWEnergetics.jl")
include("ExactRadialSpectrum.jl")

function create_grid(run_directory)
    nx, Lx = get_grid_size(run_directory, "rsw")
    grid = TwoDGrid(; nx, Lx)
    return grid
end

function start!(run_directory, tag; force_recompute=false)
    grid = create_grid(run_directory)

    open_type = force_recompute ? "w" : "a+"
    data_file = jldopen(@sprintf("%s/plot_data.jld2", run_directory), open_type)
    create_data_if_empty(run_directory, data_file, grid; force_recompute)
    close(data_file)

    data_file = jldopen(@sprintf("%s/plot_data.jld2", run_directory), "r")
    
    fig_opts = (; fontsize=24, fonts = (; regular = "Dejavu"))
        
    plot_data(run_directory, data_file, grid, tag, fig_opts)

    close(data_file)
end

function create_data_if_empty(run_directory, data_file, grid; force_recompute)
    create_flux_data(run_directory, data_file, grid; force_recompute)
    create_energy_data(run_directory, data_file, grid; force_recompute)
    create_computed_data(run_directory, data_file, grid; force_recompute=false)
end

function plot_data(run_directory, data_file, grid, tag, fig_opts)
    plot_diagnostic_energetics(run_directory, data_file, grid, tag, (; size=(1000, 600), fig_opts))
    plot_radial_energetics(run_directory, data_file, grid, tag, (; size=(600, 500), fig_opts))
    plot_spectral_fluxes(run_directory, data_file, grid, tag, (; size=(600, 500), fig_opts))
end

function create_flux_data(run_directory, data_file, grid; force_recompute=false)
    if !haskey(data_file, "flux") || force_recompute
        flux_data = compute_transfer_function(run_directory, grid)
        times, Eh_flux, Egggh_flux, Eggwh_flux, Egwwh_flux, Ewwwh_flux, Zh_flux, Zgggh_flux, Zggwh_flux, Zgwwh_flux, Zwwwh_flux = flux_data
        data_file["flux/t"] = times
        
        data_file["flux/energy_flux"] = Eh_flux
        data_file["flux/energy_flux_ggg"] = Egggh_flux
        data_file["flux/energy_flux_ggw"] = Eggwh_flux
        data_file["flux/energy_flux_gww"] = Egwwh_flux
        data_file["flux/energy_flux_www"] = Ewwwh_flux
        
        data_file["flux/enstrophy_flux"] = Zh_flux
        data_file["flux/enstrophy_flux_ggg"] = Zgggh_flux
        data_file["flux/enstrophy_flux_ggw"] = Zggwh_flux
        data_file["flux/enstrophy_flux_gww"] = Zgwwh_flux
        data_file["flux/enstrophy_flux_www"] = Zwwwh_flux
    end
end

function create_energy_data(run_directory, data_file, grid; force_recompute=false)
    if !haskey(data_file, "energy") || force_recompute
        energy_data = compute_energy_data(run_directory, grid)
        times, Egh, Ewh, KEh, KEgh, KEwh, APEh, APEgh, APEwh, Zh, KE_total, KEg_total, KEw_total, APE_total, APEg_total, APEw_total, Z_total = energy_data

        data_file["energy/t"] = times

        data_file["energy/E/geo"] = Egh
        data_file["energy/E/wave"] = Ewh
        data_file["energy/KE/total"] = KEh
        data_file["energy/KE/geo"] = KEgh
        data_file["energy/KE/wave"] = KEwh
        data_file["energy/APE/total"] = APEh
        data_file["energy/APE/geo"] = APEgh
        data_file["energy/APE/wave"] = APEwh
        data_file["energy/Z"] = Zh

        data_file["energy/series/KE/total"] = KE_total
        data_file["energy/series/KE/geo"] = KEg_total
        data_file["energy/series/KE/wave"] = KEw_total
        data_file["energy/series/APE/total"] = APE_total
        data_file["energy/series/APE/geo"] = APEg_total
        data_file["energy/series/APE/wave"] = APEw_total
        data_file["energy/series/Z"] = Z_total
    end
end

function create_computed_data(run_directory, data_file, grid; force_recompute=false)
    if !haskey(data_file, "computed") || force_recompute

        f, Cg2 = read_rsw_params(run_directory)
        Kd2 = f^2/Cg2

        data_file["computed/eddy_scale/total"] = @. sqrt(Kd2 * data_file["energy/KE/total"] / data_file["energy/PE/total"])
        data_file["computed/eddy_scale/geo"]   = @. sqrt(Kd2 * data_file["energy/KE/geo"] / data_file["energy/PE/geo"])
        
        data_file["computed/rms_u/total"] = @. sqrt(2 * data_file["energy/KE/total"])
        data_file["computed/rms_u/geo"]   = @. sqrt(2 * data_file["energy/KE/geo"])
        data_file["computed/rms_u/wave"]  = @. sqrt(2 * data_file["energy/KE/wave"])

        data_file["computed/Fr/total"] = @. data_file["computed/rms_u/total"] / sqrt(Cg2)
        data_file["computed/Fr/geo"]   = @. data_file["computed/rms_u/geo"] / sqrt(Cg2)

        data_file["computed/turnover_time/total"] = @. data_file["computed/rms_u/total"] / data_file["computed/eddy_scale/total"]
        data_file["computed/turnover_time/geo"] = @. data_file["computed/rms_u/geo"] / data_file["computed/eddy_scale/geo"]

        data_file["computed/Ro/total"] = @. data_file["computed/rms_u/total"] * data_file["computed/eddy_scale/total"] / f
        data_file["computed/Ro/geo"] = @. data_file["computed/rms_u/geo"] * data_file["computed/eddy_scale/geo"] / f
    end
end

function plot_time_series(ax, time, series; plot_options...)
    lines!(ax, time, series; plot_options...)
end

function plot_radial_data(ax, radii, weight_matrix, spectral_field; plot_options...)
    spectrum = radialspectrum(spectral_field, weight_matrix)
    spectrum = replace(spectrum, 0.0=>eps(0.0))
    lines!(ax, radii, spectrum; plot_options...)
end

function plot_radial_power_law(ax, radii, weight_matrix, spectral_field, crossing_value, power, start_radius, end_radius; plot_options...)
    spectrum = radialspectrum(spectral_field, weight_matrix)
    spectrum = replace(spectrum, 0.0=>eps(0.0))
    radii_idx = findfirst(radius -> radius >= crossing_value, radii)
    spectral_value = spectrum[radii_idx]

    dx = (end_radius - start_radius)/100
    k_vals = start_radius:dx:end_radius
    spectral_vals = spectral_value * k_vals.^(power) / (crossing_value)^power
    lines!(ax, k_vals, spectral_vals; plot_options...)
end

function plot_radial_data_integral(ax, radii, weight_matrix, spectral_field; plot_options...)
    dr = radii[2] - radii[1]
    spectrum = radialspectrum(spectral_field, weight_matrix)
    integral = cumsum((dr .* spectrum)[end:-1:1])[end:-1:1]
    lines!(ax, radii, integral; plot_options...)
end

function plot_spectral_fluxes(run_directory, data_file, grid, tag, fig_opts)
    # a. Plot total spectral flux
    # b. Plot component-wise spectral fluxes
    
    Nsnapshots = length(data_file["energy/t"])
    norm_factor = grid.Lx * grid.Ly / (grid.nx^2 * grid.ny^2) / Nsnapshots
    f, Cg2 = read_rsw_params(run_directory)
    Kd = f/sqrt(Cg2)

    radii, weight_matrix = create_radialspectrum_weights(grid, 3);
    dr = (radii[2] - radii[1])/Kd
    test_spectrum = radialspectrum(norm_factor * data_file["flux/energy_flux"], weight_matrix)
    test_integral = cumsum((dr .* test_spectrum)[end:-1:1])[end:-1:1]
    max_spec = maximum(abs.(test_integral))

    ax_opts = (; xlabel="K/Kd", xscale=log10, xticks=[1, 10, 100], limits=((radii[1]/Kd, radii[end]/Kd), (-1.1*max_spec, 1.1*max_spec)))
    
    total_flux_fig = Figure(; fig_opts...)
    total_flux_ax = Axis(total_flux_fig[1,1]; title = "RSW Spectral energy flux", ax_opts...)

    comp_flux_fig = Figure(; fig_opts...)
    comp_flux_ax = Axis(comp_flux_fig[1,1]; title = "RSW Spectral energy flux components", ax_opts...)
    
   
    plot_radial_data_integral(total_flux_ax, radii/Kd, weight_matrix, norm_factor * data_file["flux/energy_flux"], label="Π(k)", color=:black)
    
    plot_radial_data_integral(comp_flux_ax, radii/Kd, weight_matrix, norm_factor * data_file["flux/energy_flux_ggg"], label="Πggg(k)", color=:tomato)
    plot_radial_data_integral(comp_flux_ax, radii/Kd, weight_matrix, norm_factor * data_file["flux/energy_flux_ggw"], label="Πggw(k)", color=:indianred)
    plot_radial_data_integral(comp_flux_ax, radii/Kd, weight_matrix, norm_factor * data_file["flux/energy_flux_gww"], label="Πgww(k)", color=:azure4)
    plot_radial_data_integral(comp_flux_ax, radii/Kd, weight_matrix, norm_factor * data_file["flux/energy_flux_www"], label="Πwww(k)", color=:steelblue)

    Legend(total_flux_fig[2,1], total_flux_ax, orientation=:horizontal)
    Legend(comp_flux_fig[2,1], comp_flux_ax, orientation=:horizontal)

    save(@sprintf("images/%s_flux_total.png", tag), total_flux_fig)
    save(@sprintf("images/%s_flux_components.png", tag), comp_flux_fig)
end

function plot_energetics(run_directory, data_file, grid, tag, fig_opts)
    # 1. Plot time series for KE, PE, and total energy. Include diagnostics if they are available
    # 2. Plot radial energy for wave and geostrophic components, both initial and averaged

    plot_diagnostic_energetics(run_directory, data_file, grid, tag, fig_opts)
    plot_radial_energy(run_directory, data_file, grid, tag, fig_opts)
end

function plot_diagnostic_energetics(run_directory, data_file, grid, tag, fig_opts)
    times = data_file["energy/t"]
    max_time = times[end]
    f, Cg2 = read_rsw_params(run_directory)
    
    axis_opts = (; xlabel="sim. time/f", ylabel="Domain averaged energy")

    E_time_series_figure = Figure(; fig_opts...)
    E_time_series_ax = Axis(E_time_series_figure[1,1]; title="Total energy", axis_opts...)
    
    KE_PE_time_series_figure = Figure(; fig_opts...)
    KE_PE_time_series_ax = Axis(KE_PE_time_series_figure[1,1]; title="Energy components", axis_opts...)

    geo_time_series_figure = Figure(; fig_opts...)
    geo_time_series_ax = Axis(geo_time_series_figure[1,1]; title="Geostrophic energy", axis_opts...)

    wave_time_series_figure = Figure(; fig_opts...)
    wave_time_series_ax = Axis(wave_time_series_figure[1,1]; title="Wave energy", axis_opts...)
    
    if isfile(@sprintf("%s/diagnostics.jld2", run_directory))
        diag_plot_opts = (; linewidth = 1, linestyle=:solid)
        diagnostic_file = jldopen(@sprintf("%s/diagnostics.jld2", run_directory), "r")
        if(haskey(diagnostic_file["diagnostics"], "KE"))
            t_diag  = diagnostic_file["diagnostics/KE/t"]
            KE_diag = diagnostic_file["diagnostics/KE/data"]
            PE_diag = 0.5 * diagnostic_file["diagnostics/PE/data"]
            
            plot_time_series(E_time_series_ax,     t_diag/f, KE_diag + PE_diag; color=(:black,  0.5), diag_plot_opts...)
            plot_time_series(KE_PE_time_series_ax, t_diag/f, KE_diag;           color=(:maroon, 0.5), diag_plot_opts...)
            plot_time_series(KE_PE_time_series_ax, t_diag/f, PE_diag;           color=(:navy,   0.5), diag_plot_opts...)
        end
    end
    plot_opts = (;linewidth=1, linestyle=:solid)

    KE    = data_file["energy/series/KE/total"]
    KEg   = data_file["energy/series/KE/geo"] 
    KEw   = data_file["energy/series/KE/wave"]
    PE    = data_file["energy/series/APE/total"]
    PEg   = data_file["energy/series/APE/geo"]
    PEw   = data_file["energy/series/APE/wave"]

    plot_time_series(E_time_series_ax,     times/f, KE + PE; color=:black,  label="E",       plot_opts...)
    
    plot_time_series(KE_PE_time_series_ax, times/f, KE            ; color=:maroon, label="KE",      plot_opts...)
    plot_time_series(KE_PE_time_series_ax, times/f,             PE; color=:navy,   label="APE",     plot_opts...)

    plot_time_series(geo_time_series_ax,  times/f, KEg;                    color=:maroon, label="geo KE",   plot_opts...)
    plot_time_series(geo_time_series_ax,  times/f,              PEg;       color=:navy,   label="geo APE",  plot_opts...)
    plot_time_series(geo_time_series_ax,  times/f, KEg +        PEg;       color=:black,  label="geo E",    plot_opts...)
            
    plot_time_series(wave_time_series_ax, times/f,        KEw;             color=:maroon, label="wave KE",  plot_opts...)
    plot_time_series(wave_time_series_ax, times/f,                    PEw; color=:navy,   label="wave APE", plot_opts...)
    plot_time_series(wave_time_series_ax, times/f,        KEw +       PEw; color=:black,  label="wave E",   plot_opts...)

    Legend(E_time_series_figure[2, 1],     E_time_series_ax;     orientation=:horizontal)
    Legend(KE_PE_time_series_figure[2, 1], KE_PE_time_series_ax; orientation=:horizontal)
    Legend(wave_time_series_figure[2, 1],  wave_time_series_ax;  orientation=:horizontal)
    Legend(geo_time_series_figure[2, 1],   geo_time_series_ax;   orientation=:horizontal)
    
    save(@sprintf("images/%s_energy_total.png", tag), E_time_series_figure)
    save(@sprintf("images/%s_energy_KE_PE.png",   tag), KE_PE_time_series_figure)
    save(@sprintf("images/%s_energy_wave.png",  tag), wave_time_series_figure)
    save(@sprintf("images/%s_energy_geo.png",   tag), geo_time_series_figure)
end

function plot_radial_energetics(run_directory, data_file, grid, tag, fig_opts)
    Nsnapshots = count_key_snapshots(run_directory, "rsw")
    
    _, ic_rsw = load_key_snapshot(run_directory, "rsw", 1)
    _, fin_rsw = load_key_snapshot(run_directory, "rsw", Nsnapshots)
    
    dealias!(ic_rsw, grid)
    
    f, Cg2 = read_rsw_params(run_directory)
    params = (; f, Cg2)
    Kd = f/sqrt(Cg2)
    bases = compute_balanced_wave_bases(grid, params)
    ((ic_KE, ic_PE, ic_KEg, ic_PEg, ic_KEw, ic_PEw, ic_Eg, ic_Ew, Z), _) = compute_energy(ic_rsw, bases, grid, params)
    ((fin_KE, fin_PE, fin_KEg, fin_PEg, fin_KEw, fin_PEw, fin_Eg, fin_Ew, Z), _) = compute_energy(fin_rsw, bases, grid, params)
    
    radii, weight_matrix = create_radialspectrum_weights(grid, 3);

    
    ave_energy_fig = Figure(; fig_opts...)
    ic_energy_fig  = Figure(; fig_opts...)
    fin_energy_fig = Figure(; fig_opts...)

    max_E = maximum(data_file["energy/series/KE/total"]) + maximum(data_file["energy/series/APE/total"])

    ytickvals=10. .^ (-12:2)
    yticklabels = [rich("10", superscript(@sprintf("%d", value))) for value=-12:2]
    yminorticks = 10. .^ ((-12:2) .+ 0.5)
    xminorticks = 10. .^ (0.25:0.25:3)
    ax_opts = (; xscale=log10, yscale=log10, 
        xlabel="K/Kd", xticks=[1, 10, 100, 1000], 
        yticks=(ytickvals, yticklabels),
        xminorticks,
        xminorgridvisible=true,
        yminorticks,
        yminorgridvisible=true,
        limits=((3 * radii[1]/Kd, radii[end]/Kd), (1e-8 * max_E, 10 * max_E)))
    ave_energy_ax = Axis(ave_energy_fig[1,1]; title="Time-averaged radial Kinetic Energy", ax_opts...)
     ic_energy_ax = Axis( ic_energy_fig[1,1]; title="Initial radial Kinetic Energy", ax_opts...)
    fin_energy_ax = Axis(fin_energy_fig[1,1]; title="Final radial Kinetic Energy", ax_opts...)

    Nsnapshots = length(data_file["energy/t"])
    Eg = data_file["energy/E/geo"]    / Nsnapshots
    Ew = data_file["energy/E/wave"]   / Nsnapshots
    KEg = data_file["energy/KE/geo"]   / Nsnapshots
    KEw = data_file["energy/KE/wave"]  / Nsnapshots
    PEg = data_file["energy/APE/geo"]  / Nsnapshots
    PEw = data_file["energy/APE/wave"] / Nsnapshots

    geo_opts = (; color=:tomato, label="geo")
    wave_opts = (; color=:steelblue, label="wave")
    total_opts = (; color=:black, label="total")
    line_opts = (; linewidth=2)
    
    plot_radial_data(ave_energy_ax, radii/Kd, weight_matrix, KEg; line_opts..., geo_opts...)
    plot_radial_data(ave_energy_ax, radii/Kd, weight_matrix, KEw; line_opts..., wave_opts...)
    plot_radial_data(ave_energy_ax, radii/Kd, weight_matrix, KEw + KEg + PEg + PEw; line_opts..., total_opts...)
    plot_radial_power_law(ave_energy_ax, radii/Kd, weight_matrix, KEw, 20/Kd, -2.00, 10/Kd, 80/Kd; linestyle=:dash, color=:gray, label="-2 power law")
    plot_radial_power_law(ave_energy_ax, radii/Kd, weight_matrix, KEw, 20/Kd, -3.00, 10/Kd, 80/Kd; linestyle=:dash, color=:black, label="-3 power law")

    plot_radial_data(ic_energy_ax, radii/Kd, weight_matrix, ic_KEg; line_opts..., geo_opts...)
    plot_radial_data(ic_energy_ax, radii/Kd, weight_matrix, ic_KEw; line_opts..., wave_opts...)
    
    plot_radial_data(fin_energy_ax, radii/Kd, weight_matrix, fin_KEg; line_opts..., geo_opts...)
    plot_radial_data(fin_energy_ax, radii/Kd, weight_matrix, fin_KEw; line_opts..., wave_opts...)
    plot_radial_data(fin_energy_ax, radii/Kd, weight_matrix, fin_KEw + fin_KEg + fin_PEg + fin_PEw; line_opts..., total_opts...)
    plot_radial_power_law(fin_energy_ax, radii/Kd, weight_matrix, fin_KEw, 20/Kd, -2.00, 10/Kd, 80/Kd; linestyle=:dash, color=:gray, label="-2 power law")
    plot_radial_power_law(fin_energy_ax, radii/Kd, weight_matrix, fin_KEw, 20/Kd, -3.00, 10/Kd, 80/Kd; linestyle=:dash, color=:black, label="-3 power law")
    
    Legend(ave_energy_fig[2, 1], ave_energy_ax, orientation=:horizontal)
    Legend(ic_energy_fig[2, 1],  ic_energy_ax,  orientation=:horizontal)
    Legend(fin_energy_fig[2, 1], fin_energy_ax,  orientation=:horizontal)

    save(@sprintf("images/%s_energy_radial_average.png", tag), ave_energy_fig)
    save(@sprintf("images/%s_energy_radial_initial.png", tag), ic_energy_fig)
    save(@sprintf("images/%s_energy_radial_final.png", tag), fin_energy_fig)
end
