using JLD2
using Printf
using CairoMakie
include("load_file.jl")

function load_data(directory, packet_idxs)
    filename_func(idx) = @sprintf("%s/packets.%06d.jld2", directory, idx)
    num_files = sum([1 for file in readdir(directory) if occursin("packets.", file)])-1
    t, x, k, u = load_packet_analysis_files_collated(filename_func, 0:num_files; packet_idxs, load_velocity=true)
    return t, x, k, u
end

function create_wavenumber_spreading_plot(ax, k)
    for i=1:size(k, 2)
        lines!(ax, k[:,i,1], k[:,i,2], color="blue", alpha=4e-2, linewidth=3)
    end
    lines!(ax, k[1,:,1], k[1,:,2], color="black", linewidth=3)
end

function estimate_pdf(data, query_point, kernel, bandwidth)
    pointwise_contribution = sample -> kernel((sample - query_point)/bandwidth)
    return sum(pointwise_contribution, data)/length(data)/bandwidth
end

function gaussian_kernel(x)
    return 1/sqrt(2π)*exp(-x^2/2)
end

function epanechnikov_kernel(x)
    return (abs(x) <= 1) * 3/4. * (1-x^2)
end


function compute_plot_data(rel_freq, doppler_shift)
    abs_freq = rel_freq + doppler_shift
    
    μ_abs = sum(abs_freq) / length(abs_freq)
    σ_abs = sqrt(sum((abs_freq .- μ_abs).^2)/(length(abs_freq)-1))
    abs_bandwidth = 1.06 * σ_abs * length(rel_freq)^(-1/5) + 1e-2
    
    μ_rel = sum(rel_freq) / length(abs_freq)
    σ_rel = sqrt(sum((rel_freq .- μ_rel).^2)/(length(abs_freq)-1))
    rel_bandwidth = 1.06 * σ_rel * length(rel_freq)^(-1/5) + 1e-2
    
    query_start = minimum(rel_freq)/2
    query_end = 2*maximum(rel_freq)
    Npoints = max(100, 10000 * σ_rel)
    
    linspace = (0:(Npoints-1))/(Npoints-1)
    #query_points = @. exp(log(query_start) + log(query_end-query_start)*linspace)
    query_points = @. query_start + (query_end - query_start)*linspace

    rel_pdf = estimate_pdf.(Ref(rel_freq), query_points, gaussian_kernel, rel_bandwidth)
    abs_pdf = estimate_pdf.(Ref(abs_freq), query_points, gaussian_kernel, abs_bandwidth)
    return query_points, rel_pdf, abs_pdf
end

function create_frequency_spectrum(rel_ax, abs_ax, num_lines, t, x, k, u, f0, Cg)
    ω  = @. sqrt(f0^2 + Cg^2*(k[:,:,1]^2 + k[:,:,2]^2))
    ω₀ = @. sqrt(f0^2 + Cg^2*(k[1,:,1]^2 + k[1,:,2]^2))
    ω_abs  = @. abs.(sqrt(f0^2 + Cg^2*(k[:,:,1]^2 + k[:,:,2]^2)) + u[:,:,1]*k[:,:,1] + u[:,:,2]*k[:,:,2])
    doppler_shift = @. (u[:,:,1]*k[:,:,1] + u[:,:,2]*k[:,:,2])
        
    idxs = 1:floor(Int, size(ω,1)/num_lines):size(ω,1)
    println(length(idxs))
    for idx=idxs
        rel_data = (ω[idx, :][:]/f0)
        abs_shift = doppler_shift[idx, :][:]/f0
        query_points, rel_pdf, abs_pdf = compute_plot_data(rel_data, abs_shift)
        lines!(rel_ax, query_points, query_points .* rel_pdf, color="blue", alpha=1e-1 + 0.9*(idx/size(ω,1))^(100), linewidth=2)
        lines!(abs_ax, query_points, query_points .* abs_pdf, color="blue", alpha=1e-1 + 0.9*(idx/size(ω,1))^(100), linewidth=2)
    end
    
    lines!(rel_ax, ω₀[1]*ones(2)/f0, [1e-3, 10], color="black", linewidth=1, label="ft = 0")
    # lines!(abs_ax, ω₀[1]*ones(2)/f0, [1e-3, 10], color="black", linewidth=1, label="ft = 0")
    
    Nlines = 5
    Ntimes = length(t)
    deltaN = floor(Int, (Ntimes - 1) / (Nlines - 1))
    #ts = vcat([1], 2:4, [5])
    #ts = t[1, 1 + (Nlines-2) * deltaN * 1:(Nlines-2), Ntimes] / Nlines
    #idxs = findfirst.(eachrow(t' .> ts))
    idxs = vcat([1], 1 .+ deltaN * (1:(Nlines-2)), [Ntimes])
    colors = ["#ffcdcd", "#ff9a9a", "#ff6767", "#ff3434", "#ff0101"]
    for i=1:Nlines
        idx = idxs[i]
        rel_data = (ω[idx, :][:]/f0)
        abs_shift = doppler_shift[idx, :][:]/f0
        query_points, rel_pdf, abs_pdf = compute_plot_data(rel_data, abs_shift)
        lines!(rel_ax, query_points, query_points .* rel_pdf, linewidth=2, color=colors[i], label=@sprintf("ft=%.0f", floor((t[idx] - t[1])/10)*10))
        lines!(abs_ax, query_points, query_points .* abs_pdf, linewidth=2, color=colors[i], label=@sprintf("ft=%.0f", floor((t[idx] - t[1])/10)*10))
    end
    
    ω_trend = (1.01:0.5:100)*f0
    slope = -2.00
    GM_trend = @. 4*(ω_trend^2 + f0^2)/ω_trend^2 * (f0/ω_trend*(ω_trend^2-f0^2)^(-1/2))
    linear_trend = (ω₀[1])*(0.5)^(slope)*ω_trend.^(slope)
    
    lines!(rel_ax, ω_trend/f0, GM_trend, color="gray", linestyle=:solid, label="GM Spectrum"; linewidth=2)
    lines!(rel_ax, ω_trend/f0, linear_trend, color="black", linestyle=:dash, label=rich("ω", superscript(@sprintf("%.2f", slope))); linewidth=2)
    lines!(abs_ax, ω_trend/f0, GM_trend, color="gray", linestyle=:solid, label="GM Spectrum"; linewidth=2)
    lines!(abs_ax, ω_trend/f0, linear_trend, color="black", linestyle=:dash, label=rich("ω", superscript(@sprintf("%.2f", slope))); linewidth=2)
    
    axislegend(rel_ax)
    axislegend(abs_ax)
end

function create_plots(directory, image_tag, num_lines, packet_idxs, f0, Cg)
    t, x, k, u = load_data(directory, packet_idxs)

    fig = Figure(size=(600,600), fontsize=36)
    ax = Axis(fig[1,1]; title = @sprintf("Wavenumber spread, Cg=%.2f, f=%.2f", Cg, f0), limits=((-200, 200), (-200, 200)))
    image_name = @sprintf("images/wavenumber_spread_%s.png", image_tag)
    println("Creating ", image_name)
    create_wavenumber_spreading_plot(ax, k)
    save(image_name, fig)

    relative_freq_fig = Figure(size=(1000, 600), fontsize=24)
    relative_ax = Axis(relative_freq_fig[1,1]; 
            xscale=log10, yscale=log10, xticks=[1, 10, 100], 
            yticks=[1e-2, 1e-1, 1, 10, 100], limits=((1, 200), (1e-3, 10)),
            title = @sprintf("Energy spectrum (relative freq), Cg=%.2f, f=%.2f", Cg, f0), xlabel = "ω/f", ylabel = "Energy",
            xminorticks = IntervalsBetween(10), xminorticksvisible=true,
            yminorticks = IntervalsBetween(10), yminorticksvisible=true)

    abs_freq_fig = Figure(size=(1000, 600), fontsize=24)
    abs_ax = Axis(abs_freq_fig[1,1]; 
            xscale=log10, yscale=log10, xticks=[1, 10, 100], 
            yticks=[1e-2, 1e-1, 1, 10, 100], limits=((1, 200), (1e-3, 10)),
            title = @sprintf("Energy spectrum (absolute freq), Cg=%.2f, f=%.2f", Cg, f0), xlabel = "ω/f", ylabel = "Energy",
            xminorticks = IntervalsBetween(10), xminorticksvisible=true,
            yminorticks = IntervalsBetween(10), yminorticksvisible=true)

    println("Creating frequency spectra")
    create_frequency_spectrum(relative_ax, abs_ax, num_lines, t, x, k, u, f0, Cg)
    save(@sprintf("images/relative_energy_%s.png", image_tag), relative_freq_fig)
    save(@sprintf("images/absolute_energy_%s.png", image_tag), abs_freq_fig)
    println("Done")
end

function start!()
    directory = ARGS[1]
    file_tag = ARGS[2]
    create_plots(directory, file_tag)
end