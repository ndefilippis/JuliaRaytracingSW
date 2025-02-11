using AbstractFFTs
using FourierFlows
using Printf
using JLD2
using Printf
using LinearAlgebra: ldiv!
include("../RSWUtils.jl")

function hann(L)
    ell = L + 1
    N = ell - 1
    n = 0:N
    w = @. 0.5 * (1 - cos.(2*π*n/N))
    return w[1:end-1]
end

function set_up_grid(directory)
    setup_file = jldopen(@sprintf("%s/rsw.%06d.jld2", directory, 0), "r")
    nx = setup_file["grid/nx"]
    ny = setup_file["grid/ny"]
    println(nx)
    Lx = setup_file["grid/Lx"]
    Ly = setup_file["grid/Ly"]
    close(setup_file)
    grid = TwoDGrid(CPU(); nx, Lx, ny, Ly, aliased_fraction=0, T=Float64)
    return grid
end

function set_up_grid(params)
    setup_file = jldopen(@sprintf("%s/rsw.%06d.jld2", directory, 0), "r")
    f = setup_file["params/f"]
    Cg = setup_file["params/Cg"]
    close(setup_file)
    return (; f, Cg)
end

function get_total_frames(directory, file_indices)
    total_frames = 0
    
    file_list = file_indices
    for i=file_list
        file = jldopen(@sprintf("%s/rsw.%06d.jld2", directory, i))
        println(file)
        frames = keys(file["snapshots/t"])
        total_frames += length(frames)
        close(file)
    end
    return total_frames
end

function write_fourier_data(directory, file_indices, k_idx)
    grid = set_up_grid(directory)
    params = set_up_params(directory)
    file_list = file_indices
    total_frames = get_total_frames(directory, file_indices)
    window = hann(total_frames)

    println("Starting...")
    flush(stdout)
    #output_file = jldopen("radial_data.jld2", "w")
    kr = grid.kr
    dk = grid.kr[2] - grid.kr[1]
    t = zeros(total_frames)
    output_file = jldopen(@sprintf("radial_data_k=%03d.jld2", k_idx), "w")
    
    ut = zeros(Complex{Float64}, total_frames, grid.nl)
    vt = zeros(Complex{Float64}, total_frames, grid.nl)
    ηt = zeros(Complex{Float64}, total_frames, grid.nl)
    ug = zeros(Complex{Float64}, total_frames, grid.nl)
    vg = zeros(Complex{Float64}, total_frames, grid.nl)
    ηg = zeros(Complex{Float64}, total_frames, grid.nl)
    uw = zeros(Complex{Float64}, total_frames, grid.nl)
    vw = zeros(Complex{Float64}, total_frames, grid.nl)
    ηw = zeros(Complex{Float64}, total_frames, grid.nl)
    C1 = zeros(Complex{Float64}, total_frames, grid.nl)
    C2 = zeros(Complex{Float64}, total_frames, grid.nl)
    C3 = zeros(Complex{Float64}, total_frames, grid.nl)

    Φ₀, Φ₊, Φ₋ = compute_balanced_wave_bases(grid, params)
    
    println("k=" * string(k_idx))
    flush(stdout)
    base_index = 0
    for i=file_list
        file = jldopen(@sprintf("%s/rsw.%06d.jld2", directory, i), "r")
        frames = keys(file["snapshots/t"])
        for frame_idx=1:length(frames)
            frame_key = frames[frame_idx]
            t[base_index + frame_idx] = file["snapshots/t/" * frame_key]
            snapshot = file["snapshots/sol/" * frame_key]

            uh = @views snapshot[:,:,1]
            vh = @views snapshot[:,:,2]
            ηh = @views snapshot[:,:,3]
            (ugh, vgh, ηgh), (uwh, vwh, ηwh) = wave_balanced_decomposition(uh, vh, ηh, grid, params)
            c₀, c₊, c₋ = compute_balanced_wave_weights(uh, vh, ηh, Φ₀, Φ₊, Φ₋)

            ut[base_index+frame_idx,:] .= @views  uh[k_idx, :]
            vt[base_index+frame_idx,:] .= @views  vh[k_idx, :]
            ηt[base_index+frame_idx,:] .= @views  ηh[k_idx, :]
            ug[base_index+frame_idx,:] .= @views ugh[k_idx, :]
            vg[base_index+frame_idx,:] .= @views vgh[k_idx, :]
            ηg[base_index+frame_idx,:] .= @views ηgh[k_idx, :]
            uw[base_index+frame_idx,:] .= @views uwh[k_idx, :]
            vw[base_index+frame_idx,:] .= @views vwh[k_idx, :]
            ηw[base_index+frame_idx,:] .= @views ηwh[k_idx, :]
            C₀[base_index+frame_idx,:] .= @views  c₀[k_idx, :]
            C₊[base_index+frame_idx,:] .= @views  c₊[k_idx, :]
            C₋[base_index+frame_idx,:] .= @views  c₋[k_idx, :]
        end
        close(file)
        base_index += length(frames)
    end
    output_file["k"] = grid.kr[k_idx]
    output_file["t"] = t
    output_file["ut"]  = fft(window .* ut, 1)
    output_file["vt"]  = fft(window .* vt, 1)
    output_file["ηt"]  = fft(window .* ηt, 1)
    output_file["ugt"] = fft(window .* ug, 1)
    output_file["vgt"] = fft(window .* vg, 1)
    output_file["ηgt"] = fft(window .* ηg, 1)
    output_file["uwt"] = fft(window .* uw, 1)
    output_file["vwt"] = fft(window .* vw, 1)
    output_file["ηwt"] = fft(window .* ηw, 1)
    output_file["c0t"] = fft(window .* C₀, 1)
    output_file["c+t"] = fft(window .* C₊, 1)
    output_file["c-t"] = fft(window .* C₋, 1)
    close(output_file)
    println("Done with k="*string(k_idx))
end

function start!()
    job_id = parse(Int, ARGS[3])
    job_size = parse(Int, ARGS[4])
    k0 = (job_id - 1) * job_size + 1
    k1 = job_id * job_size
    println(k0, " ", k1)
    
    for k = k0:k1
        println("Writing for k = ", k)
        write_fourier_data(ARGS[1], 0:parse(Int, ARGS[2]), k)
        println("Done!")
    end
end

start!()
