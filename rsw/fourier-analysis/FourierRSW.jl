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
    #ug = zeros(Complex{Float64}, total_frames, grid.nl)
    #vg = zeros(Complex{Float64}, total_frames, grid.nl)
    #uw = zeros(Complex{Float64}, total_frames, grid.nl)
    #vw = zeros(Complex{Float64}, total_frames, grid.nl)
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

            ut[base_index+frame_idx,:] = snapshot[k_idx,:,1]
            vt[base_index+frame_idx,:] = snapshot[k_idx,:,2]

            #Gh, Wh = decompose_balanced_wave(normalized_fft, grid)

            #ug[base_index+frame_idx,:] = Gh[k_idx,:,1]
            #vg[base_index+frame_idx,:] = Gh[k_idx,:,2]
                
            #uw[base_index+frame_idx,:] = Wh[k_idx,:, 1]
            #vw[base_index+frame_idx,:] = Wh[k_idx,:, 2]
        end
        close(file)
        base_index += length(frames)
    end
    output_file["k"] = grid.kr[k_idx]
    output_file["t"] = t
    output_file["ut_series"] = ut
    output_file["vt_series"] = vt
    #output_file["ug_series"] = ug
    #output_file["vg_series"] = vg
    #output_file["uw_series"] = uw
    #output_file["vw_series"] = vw
    output_file["ut"] = fft(window .* ut, 1)
    output_file["vt"] = fft(window .* vt, 1)
    #output_file["ug"] = fft(window .* ug, 1)
    #output_file["vg"] = fft(window .* vg, 1)
    #output_file["uw"] = fft(window .* uw, 1)
    #output_file["vw"] = fft(window .* vw, 1)
    #output_file["U_balanced"] = fft(window .* ((ut + ug) + 1im*(vt + vg)), 1)
    #output_file["U_wave"] = fft(window .* (uw + 1im*vw), 1)
    #output_file["U_total"] = fft(window .* ((uw + ug + ut) + 1im*(vw + vg + vt)), 1)
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
