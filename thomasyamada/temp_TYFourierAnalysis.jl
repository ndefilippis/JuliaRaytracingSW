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
    file = jldopen("/home/nad9961/JuliaRaytracingSW/data/thomasyamada_data/long-run2/ty.jld2.00000000.jld2", "r")
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
        file = jldopen(@sprintf("/home/nad9961/JuliaRaytracingSW/data/thomasyamada_data/long-run2/ty.jld2.%08d.jld2", i))
        frames = keys(file["snapshots/t"])
        total_frames += length(frames)
        close(file)
    end
    return total_frames
end

function write_radial_fourier_data(file_indices)
    grid = set_up_grid()
    file_list = file_indices
    total_frames = get_total_frames(file_indices)
    window = hann(total_frames)

    #output_file = jldopen("radial_data.jld2", "w")
    kmax = 128
    kr = grid.kr
    dk = grid.kr[2] - grid.kr[1]
    @Threads.threads for k_idx=1:kmax
        t = zeros(total_frames)
        output_file = jldopen(@sprintf("radial_data_k=%04d.jld2", k_idx), "w")
        k_center = grid.kr[k_idx]
        radial_wave_ke = zeros(total_frames)
        rotary_wave = zeros(Complex{Float64}, total_frames)
        println("k=" * string(k_idx))
        println("filter_size=" * string(sum(k_filter)))
        flush(stdout)
        base_index = 0
        for i=file_list
            file = jldopen(@sprintf("/home/nad9961/JuliaRaytracingSW/data/thomasyamada_data/long-run2/ty.jld2.%08d.jld2", i), "r")
            frames = keys(file["snapshots/t"])
            for frame_idx=1:length(frames)
                frame_key = frames[frame_idx]
                t[frame_idx] = file["snapshots/t/" * frame_key]
                snapshot = file["snapshots/sol/" * frame_key]
                rotary_wave[base_index + frame_idx] = Wh[k_idx, k_idx, 1] + 1im * Wh[k_idx, k_idx, 2]
                radial_wave_ke[base_index + frame_idx] = abs2.(Wh[k_idx, k_idx, 1]) + abs2.(Wh[k_idx, k_idx, 2])
            end
            close(file)
            base_index += length(frames)
        end
        rotary_radial = fft(window .* rotary_wave, 1)
        wave_radial_ke = fft(window .* radial_wave_ke, 1)
        output_file["k"] = k_filter
        output_file["t"] = t
        output_file["radial_rotary"] = rotary_radial
        output_file["radial_energy"] = wave_radial_ke
        close(output_file)
    end
end

write_radial_fourier_data(25:251)
