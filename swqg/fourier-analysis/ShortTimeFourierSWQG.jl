using AbstractFFTs
using FourierFlows
using Printf
using JLD2
using Printf
using LinearAlgebra: ldiv!

function hann(L)
    ell = L + 1
    N = ell - 1
    n = 0:N
    w = @. 0.5 * (1 - cos.(2*π*n/N))
    return w[1:end-1]
end

function demean(data)
    mean = sum(data, dims=1) / size(data, 1)
    return data .- mean
end

function linear_least_squares(t, data)
    tsum = sum(t)
    t2sum = sum(t.^2)
    txsum = sum(t .* data, dims=1)
    N = size(t,1)
    slope = (N * (txsum)) / (N * t2sum - tsum.^2)
    intercept = -slope * tsum / N
    return (slope, intercept)
end

function detrend(t, data)
    m, b = linear_least_squares(t, demean(data))
    return data .- m .* t .- b
end

function clean_fft(t, data, window)
    clean_data = detrend(t, data)
    return fft(window .* clean_data, 1)
end

function set_up_grid(directory)
    setup_file = jldopen(@sprintf("%s/qgsw.%06d.jld2", directory, 0), "r")
    nx = setup_file["grid/nx"]
    ny = setup_file["grid/ny"]
    println(nx)
    Lx = setup_file["grid/Lx"]
    Ly = setup_file["grid/Ly"]
    close(setup_file)
    grid = TwoDGrid(CPU(); nx, Lx, ny, Ly, aliased_fraction=0, T=Float64)
    return grid
end

function set_up_params(directory)
    setup_file = jldopen(@sprintf("%s/qgsw.%06d.jld2", directory, 0), "r")
    Kd2 = setup_file["params/Kd2"]
    close(setup_file)
    return (; Kd2)
end

function get_total_frames(directory, file_indices)
    total_frames = 0
    
    file_list = file_indices
    for i=file_list
        file = jldopen(@sprintf("%s/qgsw.%06d.jld2", directory, i))
        println(file)
        frames = keys(file["snapshots/t"])
        total_frames += length(frames)
        close(file)
    end
    return total_frames
end

function write_fourier_data(directory, file_indices, window_length, k_idx)
    grid = set_up_grid(directory)
    params = set_up_params(directory)
    total_frames = get_total_frames(directory, file_indices) - 1 # Exclue initial time data
    window = hann(window_length)

    println("Starting...")
    flush(stdout)
    kr = grid.kr
    dk = grid.kr[2] - grid.kr[1]
    t = zeros(total_frames)
    output_file = jldopen(@sprintf("radial_data_k=%03d.jld2", k_idx), "w")
    
    ψt = zeros(Complex{Float64}, total_frames, grid.nl)

    println("k=" * string(k_idx))
    flush(stdout)
    base_index = -1 # To account for skipping the first time snap-shot
    for i=file_indices
        file = jldopen(@sprintf("%s/qgsw.%06d.jld2", directory, i), "r")
        frames = keys(file["snapshots/t"])
        for frame_idx=1:length(frames)
            if (i == file_indices[1] && frame_idx == 1)
                # Skip the initial time data
                continue
            end
            frame_key = frames[frame_idx]
            t[base_index + frame_idx] = file["snapshots/t/" * frame_key]
            qh = file["snapshots/sol/" * frame_key]
            ψh = @. -qh / (grid.Krsq + params.Kd2)
            ψt[base_index+frame_idx,:] .= @views  ψh[k_idx, :]
        end
        close(file)
        base_index += length(frames)
    end
    output_file["k"] = grid.kr[k_idx]
    for i=1:(total_frames-window_length+1)
        chopped_data = @views ψt[i:(i+window_length-1), :]
        chopped_t = @views t[i:(i+window_length-1)]
        output_file["t/" * string(i)] = t
        output_file["ψt/" * string(i)]  = clean_fft(chopped_t, chopped_data, window)
    end
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
        write_fourier_data(ARGS[1], 0:parse(Int, ARGS[2]), parse(Int, ARGS[5]), k)
        println("Done!")
    end
end

start!()
