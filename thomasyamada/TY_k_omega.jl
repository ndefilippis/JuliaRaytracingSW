include("TYUtils.jl")

using AbstractFFTs
using FourierFlows
using Printf
using JLD2
using Printf
using LinearAlgebra: ldiv!
using .TYUtils: decompose_balanced_wave, compute_balanced_basis, compute_wave_bases

function hann(L)
    ell = L + 1
    N = ell - 1
    n = 0:N
    w = @. 0.5 * (1 - cos.(2*π*n/N))
    return w[1:end-1]
end

function set_up_grid()
    file = jldopen("/vast/nad9961/thomasyamada_simulation/50459728/ty.jld2.00000000.jld2", "r")
    frames = keys(file["snapshots/sol"])
    t = file["snapshots/t"][frames[end]]
    solution = file["snapshots/sol"][frames[end]]
    nx = file["grid/nx"]
    ny = file["grid/ny"]
    Lx = file["grid/Lx"]
    Ly = file["grid/Ly"]
    close(file)
    grid = TwoDGrid(CPU(); nx, Lx, ny, Ly, aliased_fraction=0, T=Float64)
    return grid
end

function get_total_frames(file_indices)
    total_frames = 0
    
    file_list = file_indices
    for i=file_list
        file = jldopen(@sprintf("/vast/nad9961/thomasyamada_simulation/50459728/ty.jld2.%08d.jld2", i))
        frames = keys(file["snapshots/t"])
        total_frames += length(frames)
        close(file)
    end
    return total_frames
end

function write_fourier_data(file_indices, k_idx)
    grid = set_up_grid()
    file_list = file_indices
    total_frames = get_total_frames(file_indices)
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
    ug = zeros(Complex{Float64}, total_frames, grid.nl)
    vg = zeros(Complex{Float64}, total_frames, grid.nl)
    uw = zeros(Complex{Float64}, total_frames, grid.nl)
    vw = zeros(Complex{Float64}, total_frames, grid.nl)
    println("k=" * string(k_idx))
    flush(stdout)
    base_index = 0
    for i=file_list
        file = jldopen(@sprintf("/vast/nad9961/thomasyamada_simulation/50459728/ty.jld2.%08d.jld2", i), "r")
        frames = keys(file["snapshots/t"])
        for frame_idx=1:length(frames)
            frame_key = frames[frame_idx]
            t[base_index + frame_idx] = file["snapshots/t/" * frame_key]
            snapshot = file["snapshots/sol/" * frame_key]
            normalized_fft = snapshot#rfft(irfft(snapshot, grid.nx, (1,2)), (1,2))

            ut[base_index+frame_idx,:] = (-1im * grid.l  .* normalized_fft)[k_idx,:,1]
            vt[base_index+frame_idx,:] = ( 1im * grid.kr .* normalized_fft)[k_idx,:,1]

            Gh, Wh = decompose_balanced_wave(normalized_fft, grid)

            ug[base_index+frame_idx,:] = Gh[k_idx,:,1]
            vg[base_index+frame_idx,:] = Gh[k_idx,:,2]
                
            uw[base_index+frame_idx,:] = Wh[k_idx,:, 1]
            vw[base_index+frame_idx,:] = Wh[k_idx,:, 2]
        end
        close(file)
        base_index += length(frames)
    end
    output_file["k"] = grid.kr[k_idx]
    output_file["t"] = t
    output_file["ut_series"] = ut
    output_file["vt_series"] = vt
    output_file["ug_series"] = ug
    output_file["vg_series"] = vg
    output_file["uw_series"] = uw
    output_file["vw_series"] = vw
    output_file["ut"] = fft(window .* ut, 1)
    output_file["vt"] = fft(window .* vt, 1)
    output_file["ug"] = fft(window .* ug, 1)
    output_file["vg"] = fft(window .* vg, 1)
    output_file["uw"] = fft(window .* uw, 1)
    output_file["vw"] = fft(window .* vw, 1)
    output_file["U_balanced"] = fft(window .* ((ut + ug) + 1im*(vt + vg)), 1)
    output_file["U_wave"] = fft(window .* (uw + 1im*vw), 1)
    output_file["U_total"] = fft(window .* ((uw + ug + ut) + 1im*(vw + vg + vt)), 1)
    close(output_file)
    println("Done with k="*string(k_idx))
end

write_fourier_data(0:400, parse(Int, ARGS[1]))
