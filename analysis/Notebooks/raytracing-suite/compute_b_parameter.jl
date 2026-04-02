using JLD2
using FourierFlows
using CairoMakie
using AbstractFFTs
using Printf
using Interpolations
using LsqFit
include("./AnalysisUtils.jl")

function compute_flow_froude(run_directory, grid)
    Nsnaps = count_key_snapshots(run_directory, "2Lqg")
    t, sol = load_key_snapshot(run_directory, "2Lqg", Nsnaps)
    params = read_2Lqg_params(run_directory, "2Lqg")

    f0 = 3.0
    Cg = 1.0
    qh = 0.5 * (sol[:,:,1] - sol[:,:,2])
    ψh = @. -qh / (grid.Krsq + 2*params.F)
    uh = @. -1im * grid.l  * ψh
    vh = @.  1im * grid.kr * ψh
    vxh = 1im * grid.kr .* vh
    uyh = 1im * grid.l .* uh
    vx = irfft(vxh, grid.nx)
    uy = irfft(uyh, grid.nx)
    u = irfft(uh, grid.nx)
    v = irfft(vh, grid.nx)
    rms_u = sqrt.(sum(u.^2 + v.^2)/grid.nx/grid.ny)

    return rms_u/Cg
end

function compute_ψ_correlation(fourier_directory, grid)
    file = jldopen(@sprintf("%s/%d/radial_data_k=%03d.jld2", fourier_directory, 1, 1))
    ω_size = size(file["ψt"], 1)
    t = file["t"]
    T = t[end] - t[1]
    dt = t[2] - t[1]
    Nω = length(t)
    ωs = fftshift(fftfreq(Nω, 1/dt) * 2π)

    Ch = zeros(ω_size, 2 * grid.nkr)
    for file_idx=1:(grid.nkr-1)
        file = jldopen(@sprintf("%s/%d/radial_data_k=%03d.jld2", fourier_directory, floor(Int, (file_idx-1)/4 + 1), file_idx))
        k = file["k"]
        q = @. sqrt(k^2 + grid.l[:]^2)
        K_idx = floor.(Int, q) .+ 1
        ψt = file["ψt"]
        norm_factor = 1 / Nω^2 / grid.nx^2 / grid.ny^2 / 2
        # println(file_idx)
        result = real.(conj.(ψt) .* ψt) * norm_factor
        @views Ch[:, K_idx] .+= result
    end
    return ωs, Ch
end

function compute_b(Ch, ωs, grid)
    Kd = 3.0
    f0 = 3.0
    c = f0 / Kd
    
    Npoints = 176
    k_max = 176
    k = (1:Npoints)/(Npoints) * k_max
    D_11_p = zeros(Npoints)
    
    ω = @. sqrt(f0^2 + c^2*k^2)
    Cg = @. c^2 * k / ω
    
    dq = 0.1
    q = 0.0:dq:grid.kr[end]
    dη = 0.01
    η = (0.0:dη:2π)'

    Ch_shifted = fftshift(Ch[:, 1:grid.nkr], 1)
    Ch_itp = extrapolate(scale(interpolate(Ch_shifted, BSpline(Cubic())), ωs, 0:(grid.nkr-1)), 0)
    
    for idx=1:Npoints
        σ = -Cg[idx] * q .* cos.(η)
        D_11_p[idx] = k[idx]^2 * sum(q.^5 .* cos.(η).^2 .* sin.(η).^2 .* Ch_itp.(σ, q) * dq * dη)
    end
    m(k, p) = p[1]*k.^2# .+ p[2]*k# .+ p[3]
    p0 = [1e-2]
    fit = curve_fit(m, k/Kd, D_11_p, p0)
    return fit.param[1]
end

function main(run_directory, fourier_directory, output_directory)
    nx, Lx = get_grid_size(run_directory, "2Lqg")
    grid = TwoDGrid(; nx, Lx)
    println("====")
    flush(stdout)
    rms_u = compute_flow_froude(run_directory, grid)
    println(@sprintf("<u> = %0.3e", rms_u))
    flush(stdout)
    ωs, Ch = compute_ψ_correlation(fourier_directory, grid)
    b = compute_b(Ch, ωs, grid)
    println(@sprintf("b = %0.3e", b))
    flush(stdout)
    output_file = jldopen(output_directory, "w")
    output_file["b"] = b
    output_file["Fr"] = rms_u
end

main(ARGS[1], ARGS[2], ARGS[3])
