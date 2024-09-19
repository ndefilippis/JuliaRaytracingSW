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
    file = jldopen("/vast/nad9961/thomasyamada_simulation/48950250/ty.jld2.00000000.jld2", "r")
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
        file = jldopen(@sprintf("/vast/nad9961/thomasyamada_simulation/48950250/ty.jld2.%08d.jld2", i))
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
    kmax = 256
    kr = grid.kr
    dk = grid.kr[2] - grid.kr[1]
    for k_idx=1:kmax
        t = zeros(total_frames)
        output_file = jldopen(@sprintf("radial_data_k=%04d.jld2", k_idx), "w")
        k_center = grid.kr[k_idx]
        k_filter = k_center^2 .<= grid.Krsq .< (k_center + dk)^2
        radial_ut = zeros(total_frames, sum(k_filter))
        radial_vt = zeros(total_frames, sum(k_filter))
        radial_uc = zeros(total_frames, sum(k_filter))
        radial_vc = zeros(total_frames, sum(k_filter))
        radial_uw = zeros(total_frames, sum(k_filter))
        radial_vw = zeros(total_frames, sum(k_filter))
        radial_wave_ke = zeros(total_frames, sum(k_filter))
        rotary_wave = zeros(Complex{Float64}, total_frames, sum(k_filter))
        println("k=" * string(k_idx))
        println("filter_size=" * string(sum(k_filter)))
        flush(stdout)
        base_index = 0
        @Threads.threads for i=file_list
            file = jldopen(@sprintf("/home/nad9961/JuliaRaytracingSW/data/thomasyamada_data/long-run2/ty.jld2.%08d.jld2", i), "r")
            frames = keys(file["snapshots/t"])
            for frame_idx=1:length(frames)
                frame_key = frames[frame_idx]
                t[base_index + frame_idx] = file["snapshots/t/" * frame_key]
                snapshot = file["snapshots/sol/" * frame_key]
                normalized_fft = rfft(irfft(snapshot, grid.nx, (1,2)), (1,2))

                radial_ut[base_index+frame_idx, :] = (-1im * grid.l  .* normalized_fft)[k_filter,1]
                radial_vt[base_index+frame_idx, :] = ( 1im * grid.kr .* normalized_fft)[k_filter,1]
                radial_uc[base_index+frame_idx, :] = normalized_fft[k_filter,2]
                radial_vc[base_index+frame_idx, :] = normalized_fft[k_filter,3]

                Gh, Wh = decompose_balanced_wave(normalized_fft, grid)
                    
                radial_uw[base_index+frame_idx, :] = Wh[k_filter, 1]
                radial_vw[base_index+frame_idx, :] = Wh[k_filter, 2]

                radial_wave_ke[base_index + frame_idx, :] = abs2.(Wh[k_filter, 1]) + abs2.(Wh[k_filter, 2])
                rotary_wave[base_index + frame_idx, :] = Wh[k_filter, 1] + 1im * Wh[k_filter, 2]
            end
            close(file)
            base_index += length(frames)
        end
        fft_ut = fft(window .* radial_ut, 1)
        fft_vt = fft(window .* radial_vt, 1)
        fft_uc = fft(window .* radial_uc, 1)
        fft_vc = fft(window .* radial_vc, 1)
        fft_uw = fft(window .* radial_uw, 1)
        fft_vw = fft(window .* radial_vw, 1)
        rotary_radial = fft(window .* rotary_wave, 1)
        wave_radial_ke = fft(window .* radial_wave_ke, 1)
        output_file["k"] = k_center
        output_file["k_filter"] = k_filter
        output_file["t"] = t
        output_file["ut"] = sum(abs2.(fft_ut, dims=2))
        output_file["ut_test"] = sum(abs.(fft_ut, dims=2))
        output_file["vt"] = sum(abs2.(fft_vt, dims=2))
        output_file["uc"] = sum(abs2.(fft_uc, dims=2))
        output_file["vc"] = sum(abs2.(fft_vc, dims=2))
        output_file["uw"] = sum(abs2.(fft_uw, dims=2))
        output_file["vw"] = sum(abs2.(fft_vw, dims=2))
        output_file["radial_rotary"] = sum(abs2.(rotary_radial), dims=2)
        output_file["radial_energy"] = sum(abs.(wave_radial_ke), dims=2)
        close(output_file)
        println("Done with k="*string(k_idx))
    end
end

write_radial_fourier_data(25:251)
