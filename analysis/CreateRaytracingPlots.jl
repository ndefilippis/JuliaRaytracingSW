using JLD2
using CairoMakie
include("load_file.jl")

function create_filename_function(directory)
    return filename_func(idx) = @sprintf("%s/packets.%06d.jld2", idx)
end

function create_wavenumber_spread_figure(k, packet_idxs; figure_options...)
    fig = Figure(figure_options...)
    ax = Axis(fig[1, 1]; title="Wavenumber spread", xlabel="l", ylabel="l", aspect=1)
    for idx=packet_idxs
        lines!(ax, k[:,i,1], k[:,i,2], color="blue", alpha=4e-2, linewidth=3)
    end
    lines!(ax, k[1,:,1], k[1,:,2], color="black", linewidth=3)

    return fig
end

function create_absolute_frequecny_figure(k, u, t; figure_options...)
    Ω = sqrt.(1 .+ (k[:,:,1].^2 + k[:,:,2].^2)) + sum(u .* k, dims=3)[:,:,1]
    Ω = Ω[2:end, :]
    
    fig = Figure(size=(800, 600), fontsize = 36, figure_padding = 30)
    ax = Axis(fig2[1, 1]; 
            title = "Relative change in absolute frequency", xlabel="ft", ylabel="(Ω - Ω₀)/Ω₀"
            #limits = ((0, maximum(t1)), (-1., 5.))
    )
    for i=1:25
        initial_omega = Ω[1,i]
        lines!(ax2, t[2:end], (Ω[:, i] .- initial_omega) / initial_omega, linewidth=3)
    end
    
    #save("QG_absolute_frequency.png", fig2)
    fig2
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

function compute_plot_data(data)
    μ = sum(data) / length(data)
    σ = sqrt(sum((data .- μ).^2)/(length(data)-1))
    bandwidth = 1.06 * σ * length(data)^(-1/5) + 1e-2
    #println(bandwidth)
    query_start = minimum(data)/2
    query_end = 2*maximum(data)
    Npoints = max(100, 10000 * σ)
    #println(Npoints)
    linspace = (0:(Npoints-1))/(Npoints-1)
    #query_points = @. exp(log(query_start) + log(query_end-query_start)*linspace)
    query_points = @. query_start + (query_end - query_start)*linspace
    pdf = estimate_pdf.(Ref(data), query_points, gaussian_kernel, bandwidth)
    return query_points, pdf
end

function create_kde_plot()
    f = Figure(size=(1000, 600), fontsize=24)
    ax = Axis(f[1,1]; xscale=log10, yscale=log10, xticks=[1, 10], yticks=[1e-2, 1e-1, 1, 10], limits=((1, maximum(ω)), (1e-3, 10)),
     title = "Energy spectrum",
     xlabel = "ω/f",
     ylabel = "Energy",
     xminorticks = IntervalsBetween(10),
     xminorticksvisible=true,
     yminorticks = IntervalsBetween(10),
     yminorticksvisible=true)
    #bandwidths = 0.1:0.3:1.2
    #bandwidths = 0.01:0.01:0.05
    idxs = 1:size(ω,1)
    println(length(idxs))
    for idx=idxs
        data = ω_abs[idx, :][:]
        query_points, pdf = compute_plot_data(data)
        lines!(ax, query_points, query_points .* pdf, color="blue", alpha=1e-1 + 0.9*(idx/size(ω,1))^(100), linewidth=2)
        #scatter!(ax, data, 1e-2 .* ones(length(data)), marker=:vline, markersize=10)
    end
    
    lines!(ax, ω₀[1]*ones(2), [1e-3, 10], color="black", linewidth=1, label="ft = 0")  
    
    idxs = [205, 409, 613, 816]
    colors = ["#ffdede", "#ff8989", "#ff5656", "#ff0000"]
    for i=1:length(idxs)
        idx = idxs[i]
        data = ω_abs[idx, :][:]
        query_points, pdf = compute_plot_data(data)
        lines!(ax, query_points, query_points .* pdf, linewidth=2, color=colors[i], label=@sprintf("ft=%.0f", floor(t[idx]/10)*10))
    end
    
    ω_trend = 1:0.1:maximum(ω)
    slope = -2.00
    trend = ω₀[1]*(1/2.)^(slope)*ω_trend.^(slope)
    
    lines!(ax, ω_trend, trend, color="black", linestyle=:dash, label=rich("ω", superscript(@sprintf("%.2f", slope))); linewidth=2)
    axislegend(ax)
    
    f
end