matern_model(ω, p) = @. p[1] * ((ω - p[2])^2 + p[3]^2)^p[4]
log_matern_model(ω, p) = @. log(p[1] * ((ω - p[2])^2 + p[3]^2)^p[4])

function mean(data)
    return sum(data) / length(data)
end

function variance(data)
    return sum((data .- mean(data)).^2) / length(data)
end

function std(data)
    return sqrt.(variance(data))
end

function idxcounts(idxtuple, Nbins)
    bincounts = zeros(Nbins, Nbins)
    for (i,j)=idxtuple
        bincounts[i,j] += 1
    end
    return bincounts
end

function read_parameters(directory)
    file = jldopen(@sprintf("%s/packets.%06d.jld2", directory, 0))
    f0 = file["params/f0"]
    Cg = file["params/Cg"]
    close(file)
    return f0, Cg
end

function compute_strain_vorticity(ψh)
    uh =  -grid.l  .* ψh
    vh =   grid.kr .* ψh
    
    uxh =  grid.kr .* uh
    uyh =  grid.l  .* uh
    vxh =  grid.kr .* vh
    vyh =  grid.l  .* vh
    
    ζh  = vxh - uyh
    σnh = uxh - vyh
    σsh = vxh + uyh;
    
    ζ = zeros(grid.nx, grid.ny)
    σn = zeros(grid.nx, grid.ny)
    σs = zeros(grid.nx, grid.ny)
    
    ldiv!(ζ, grid.rfftplan, ζh)
    ldiv!(σn, grid.rfftplan, σnh)
    ldiv!(σs, grid.rfftplan, σsh)
    
    σ = @. sqrt(σn^2 + σs^2)

    return σs, σn, σ, ζ
end

function load_last_snapshot(directory, grid)
    filename_func(idx) = @sprintf("%s/qgsw.%06d.jld2", directory, idx)
    num_files = sum([1 for file in readdir(directory) if occursin("qgsw.", file)])-1
    first_file = jldopen(@sprintf("%s/qgsw.%06d.jld2", directory, 0))
    last_file = jldopen(@sprintf("%s/qgsw.%06d.jld2", directory, num_files))
    
    Kd2 = first_file["params/Kd2"]
    last_key = keys(last_file["snapshots/t"])[end]
    t_final = last_file["snapshots/t/" * last_key]
    qh_final = last_file["snapshots/sol/" * last_key]
    ψh_final = -qh_final ./ (grid.Krsq .+ Kd2)
    uh_final = -1im * grid.l .* ψh_final
    vh_final =  1im * grid.kr .* ψh_final
    q_final = irfft(qh_final, grid.nx)
    ψ_final = irfft(ψh_final, grid.nx)
    u_final = irfft(uh_final, grid.nx)
    v_final = irfft(vh_final, grid.nx)
    
    println(last_file["snapshots/t/" * last_key])
    close(last_file)
    return t_final, q_final, ψ_final, u_final, v_final, Kd2
end

function load_first_last_frame(directory)
    filename_func(idx) = @sprintf("%s/packets.%06d.jld2", directory, idx)
    num_files = sum([1 for file in readdir(directory) if occursin("packets.", file)])-1
    first_file = jldopen(@sprintf("%s/packets.%06d.jld2", directory, 0))
    last_file = jldopen(@sprintf("%s/packets.%06d.jld2", directory, num_files))
    first_snapshot = keys(first_file["p/t"])[1]
    last_snapshot = keys(last_file["p/t"])[end]
    t0 = first_file["p/t/" * first_snapshot]
    x0 = first_file["p/x/" * first_snapshot]
    k0 = first_file["p/k/" * first_snapshot]
    u0 = first_file["p/u/" * first_snapshot]
    t1 =  last_file["p/t/" * last_snapshot]
    x1 =  last_file["p/x/" * last_snapshot]
    k1 =  last_file["p/k/" * last_snapshot]
    u1 =  last_file["p/u/" * last_snapshot]
    close(last_file)
    close(first_file)
    return (t0, x0, k0, u0), (t1, x1, k1, u1)
end

function count_snapshots(directory)
    num_snapshots = 0
    filename_func(idx) = @sprintf("%s/packets.%06d.jld2", directory, idx)
    num_files = sum([1 for file in readdir(directory) if occursin("packets.", file)])-1
    file_idx = 1
    for j=0:num_files
        file = jldopen(filename_func(j))
        num_snapshots += length(keys(file["p/t"]))
        close(file)
    end
    return num_snapshots
end

function count_qgsw_snapshots(directory)
    num_snapshots = 0
    filename_func(idx) = @sprintf("%s/qgsw.%06d.jld2", directory, idx)
    num_files = sum([1 for file in readdir(directory) if occursin("qgsw.", file)])-1
    file_idx = 1
    for j=0:num_files
        file = jldopen(filename_func(j))
        num_snapshots += length(keys(file["snapshots/t"]))
        close(file)
    end
    return num_snapshots
end

function load_snapshot(directory, snap_idx; load_gradients=false)
    filename_func(idx) = @sprintf("%s/packets.%06d.jld2", directory, idx)
    num_files = sum([1 for file in readdir(directory) if occursin("packets.", file)])-1
    file_idx = 0
    num_snapshots = 0
    file = nothing
    file_idx = snap_idx
    for j=0:num_files
        file = jldopen(filename_func(j))
        num_snapshots += length(keys(file["p/t"]))
        if snap_idx <= num_snapshots
            break
        end
        file_idx -= length(keys(file["p/t"]))
        close(file)
    end
    snapshot = keys(file["p/t"])[file_idx]
    t =  file["p/t/" * snapshot]
    x =  file["p/x/" * snapshot]
    k =  file["p/k/" * snapshot]
    u =  file["p/u/" * snapshot]
    ux = nothing
    if(load_gradients)
        ux = file["p/g/" * snapshot]
    end
    close(file)
    if(load_gradients)
        return (t, x, k, u, ux)
    else
        return (t, x, k, u)
    end
end

function compute_U_rms(u, v)
    return sqrt.(sum(u.^2 + v.^2) / size(u, 1) / size(u, 2))
end

function compute_Cg_rms(k, f0, sqrtgH)
    ω = compute_ω(k, f0, sqrtgH)
    return sqrt.(sum((sqrtgH^2 * k ./ ω).^2) / size(k, 1))
end

function load_qgsw_snapshot(directory, grid, snap_idx)
    filename_func(idx) = @sprintf("%s/qgsw.%06d.jld2", directory, idx)
    num_files = sum([1 for file in readdir(directory) if occursin("qgsw.", file)])-1
    num_snapshots = 0
    file = nothing
    file_idx = snap_idx
    for j=0:num_files
        file = jldopen(filename_func(j))
        num_snapshots += length(keys(file["snapshots/t"]))
        if snap_idx <= num_snapshots
            break
        end
        file_idx -= length(keys(file["snapshots/t"]))
        close(file)
    end
    Kd2 = file["params/Kd2"]
    key = keys(file["snapshots/t"])[file_idx]
    t = file["snapshots/t/" * key]
    qh = file["snapshots/sol/" * key]
    ψh = -qh ./ (grid.Krsq .+ Kd2)
    uh = -1im * grid.l .* ψh
    vh =  1im * grid.kr .* ψh
    q = irfft(qh, grid.nx)
    ψ = irfft(ψh, grid.nx)
    u = irfft(uh, grid.nx)
    v = irfft(vh, grid.nx)

    close(file)
    return t, q, ψ, u, v, Kd2
end

function load_qgswh_snapshot(directory, grid, snap_idx)
    filename_func(idx) = @sprintf("%s/qgsw.%06d.jld2", directory, idx)
    num_files = sum([1 for file in readdir(directory) if occursin("qgsw.", file)])-1
    num_snapshots = 0
    file = nothing
    file_idx = snap_idx
    for j=0:num_files
        file = jldopen(filename_func(j))
        num_snapshots += length(keys(file["snapshots/t"]))
        if snap_idx <= num_snapshots
            break
        end
        file_idx -= length(keys(file["snapshots/t"]))
        close(file)
    end
    Kd2 = file["params/Kd2"]
    key = keys(file["snapshots/t"])[file_idx]
    t = file["snapshots/t/" * key]
    qh = file["snapshots/sol/" * key]
    ψh = -qh ./ (grid.Krsq .+ Kd2)
    uh = -1im * grid.l .* ψh
    vh =  1im * grid.kr .* ψh

    close(file)
    return t, qh, ψh, uh, vh, Kd2
end

function compute_ω(k, f, Cg)
    return sqrt.(f^2 .+ Cg^2*(k[:,1].^2 + k[:,2].^2))
end

function compute_ω(k, f, Cg, freq_sign)
    return freq_sign .* sqrt.(f^2 .+ Cg^2*(k[:,1].^2 + k[:,2].^2))
end

function compute_doppler_shift(k, u)
    return @. k[:,1] * u[:,1] + k[:,2] * u[:,2]
end

function compute_Ω(k, u, f, Cg)
    return compute_ω(k, f, Cg) + compute_doppler_shift(k, u)
end

function compute_Ω(k, u, f, Cg, freq_sign)
    return compute_ω(k, f, Cg, freq_sign) + compute_doppler_shift(k, u)
end