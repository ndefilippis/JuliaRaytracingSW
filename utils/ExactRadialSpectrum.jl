using FourierFlows
using SparseArrays

function create_radialspectrum_weights(grid, resolution_factor=2)
    dk = grid.kr[2] - grid.kr[1]
    radii = (1:(resolution_factor * (grid.nkr-2))) / resolution_factor * dk
    num_radii = floor(Int, resolution_factor * (grid.nkr-2)
    previous_weights = zeros(grid.nkr, grid.nl)
    weight_matrix = Vector{SparseMatrixCSC}(undef, num_radii)
    for r_idx=1:num_radii
        radius = radii[r_idx]
        weights = get_weights(grid, radius)
        true_weights = hcat(weights, weights[:, end-1:-1:2])
        true_weights[2:end-1, :] *= 2
        weight_matrix[r_idx] = sparse(true_weights - previous_weights)
        previous_weights = true_weights
    end

    return radii, weight_matrix
end

function radialspectrum(data, radial_weights)
    return [sum(data .* weight) for weight in radial_weights]
end

function get_weights(grid, radius)
    weights = zeros(grid.nkr, grid.nkr)   
    k = grid.kr
    l = (grid.l[1:grid.nkr])'
    l[grid.nkr] = -grid.l[grid.nkr]
    Krsq = k.^2 .+ l.^2
    dk = k[2] - k[1]
    dk_half = dk/2
    
    W_k = k .- dk_half
    E_k = k .+ dk_half
    S_l = l .- dk_half
    N_l = l .+ dk_half
    
    S_l[1] = 0
    N_l[1] = dk_half
    W_k[1] = 0
    E_k[1] = dk_half

    weights[radius^2 .<= (S_l.^2 .+ W_k.^2)] .= 0
    weights[radius^2 .>= (N_l.^2 .+ E_k.^2)] .= 1
    
    clip = (S_l.^2 .+ W_k.^2) .<= radius^2 .<= (N_l.^2 .+ E_k.^2)

    
    upper_octant = (W_k .<= S_l)
    NW_clip = (radius^2 .< N_l.^2 .+ W_k.^2)
    SE_clip = (radius^2 .< S_l.^2 .+ E_k.^2)
    corner_clip = upper_octant .& NW_clip .& SE_clip .& clip
    flat_clip = upper_octant .& NW_clip .& .~SE_clip .& clip
    top_clip = upper_octant .& .~ NW_clip .& .~SE_clip .& clip

    set_corner_clip_weights(weights, corner_clip, W_k, E_k, S_l, N_l, radius)
      set_flat_clip_weights(weights,   flat_clip, W_k, E_k, S_l, N_l, radius)
      set_top_clip_weights(weights,     top_clip, W_k, E_k, S_l, N_l, radius)

    return weights
end

function set_corner_clip_weights(weights, corner_clips, W_k, E_k, S_l, N_l, radius)
    W_ks = W_k[getindex.(findall(corner_clips), 1)]
    E_ks = E_k[getindex.(findall(corner_clips), 1)]
    S_ls = S_l[getindex.(findall(corner_clips), 2)]
    N_ls = N_l[getindex.(findall(corner_clips), 2)]
    W_intersection = @. sqrt(radius^2 - W_ks^2)
    S_intersection = @. sqrt(radius^2 - S_ls^2)
    
    triangle_area = @. 0.5 * (S_intersection - W_ks) * (W_intersection - S_ls)

    chord_length = @. sqrt((S_intersection - W_ks)^2 + (W_intersection - S_ls)^2)
    theta = @. 2 * asin(chord_length / 2 / radius)
    circle_area = @. radius^2 / 2 * (theta - sin(theta))

    area = triangle_area + circle_area
    weight = @. area / ((E_ks - W_ks) * (N_ls - S_ls))
    weights[corner_clips] = weight
    weights'[corner_clips] = weight
end

function set_flat_clip_weights(weights, flat_clips, W_k, E_k, S_l, N_l, radius)
    W_ks = W_k[getindex.(findall(flat_clips), 1)]
    E_ks = E_k[getindex.(findall(flat_clips), 1)]
    S_ls = S_l[getindex.(findall(flat_clips), 2)]
    N_ls = N_l[getindex.(findall(flat_clips), 2)]
    W_intersection = @. sqrt(radius^2 - W_ks^2)
    E_intersection = @. sqrt(radius^2 - E_ks^2)

    rectangle_area = @. (E_ks - W_ks) * (E_intersection - S_ls)
    
    triangle_area = @. 0.5 * (E_ks - W_ks) * (W_intersection - E_intersection)

    chord_length = @. sqrt((E_ks - W_ks)^2 + (W_intersection - E_intersection)^2)
    theta = @. 2 * asin(chord_length / 2 / radius)
    circle_area = @. radius^2 / 2 * (theta - sin(theta))
    
    area = rectangle_area + triangle_area + circle_area
    weight = @. area / ((E_ks - W_ks) * (N_ls - S_ls))
    weights[flat_clips] = weight
    weights'[flat_clips] = weight
end

function set_top_clip_weights(weights, top_clips, W_k, E_k, S_l, N_l, radius)
    W_ks = W_k[getindex.(findall(top_clips), 1)]
    E_ks = E_k[getindex.(findall(top_clips), 1)]
    S_ls = S_l[getindex.(findall(top_clips), 2)]
    N_ls = N_l[getindex.(findall(top_clips), 2)]
    N_intersection = @. sqrt(radius^2 - N_ls^2)
    E_intersection = @. sqrt(radius^2 - E_ks^2)

    W_rectangle_area = @. (N_intersection - W_ks) * (N_ls - S_ls)
    S_rectangle_area = @. (E_ks - N_intersection) * (E_intersection - S_ls)
    
    triangle_area = @. 0.5 * (E_ks - N_intersection) * (N_ls - E_intersection)

    chord_length = @. sqrt((E_ks - N_intersection)^2 + (N_ls - E_intersection)^2)
    theta = @. 2 * asin(chord_length / 2 / radius)
    circle_area = @. radius^2 / 2 * (theta - sin(theta))
    
    area = W_rectangle_area + S_rectangle_area + triangle_area + circle_area
    weight = @. area / ((E_ks - W_ks) * (N_ls - S_ls))
    weights[top_clips] = weight
    weights'[top_clips] = weight
end