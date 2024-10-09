module CUDAInterpolations

export bicubic_interpolate, bilinear_interpolate, linear_interpolate, cubic_interpolate, two_point_linear_interpolate

@inline function fractional_index(val, nodes, dx)
    frac_idx = (val - nodes[1])/dx + 1
    index = Base.unsafe_trunc(Int, frac_idx)
    αx = mod(frac_idx, 1)
    return (index, αx)
end

@inline function periodic_boundary(idx, N)
    return mod(idx-1, N) + 1
end

@inline function get_indices(query_point, nodes, dx, N)
    idx, α = fractional_index(query_point, nodes, dx)
    idx    = periodic_boundary(idx, N)
    next_idx = periodic_boundary(idx+1, N)
    return (idx, next_idx, α)
end

@inline function linear_interpolate(α, f₀, f₁)
    return α * f₁ + (1 - α) * f₀ 
end

@inline function two_point_linear_interpolate(query_point, f₀, f₁, node₀, node₁)
    α = (query_point - node₀) / (node₁ - node₀)
    return linear_interpolate(α, f₀, f₁)
end

@inline function linear_interpolate(query_point, field, nodes, dx, N)
    idx, next_idx, α = get_indices(query_point, nodes, dx, N)
    f₀ = @inbounds field[idx]
    f₁ = @inbounds field[next_idx]
    return linear_interpolate(α, f₀, f₁)
end

@inline function cubic_interpolate(α, f₀, f₁, m₀, m₁)
    return      f₀                           + 
                              m₀       * α   + 
            (-3*f₀ + 3*f₁ - 2*m₀ - m₁) * α^2 + 
            ( 2*f₀ - 2*f₁ +   m₀ + m₁) * α^3
end

@inline function cubic_interpolate(query_point, field, field_deriv, nodes, dx, N)
    idx, next_idx, α = get_indices(query_point, nodes, dx, N)
    f₀ = @inbounds field[idx]
    f₁ = @inbounds field[next_idx]
    m₀ = @inbounds field_deriv[idx] * dx
    m₁ = @inbounds field_deriv[next_idx] * dx
    return cubic_interpolate(α, f₀, f₁, m₀, m₁)
end

@inline function bilinear_interpolate(
        x_query_point, y_query_point, 
        field, 
        x_nodes, y_nodes, 
        dx, N)
    x_idx, next_x_idx, ξ = get_indices(x_query_point, x_nodes, dx, N)
    y_idx, next_y_idx, ζ = get_indices(y_query_point, y_nodes, dx, N)
    f₀₀ = @inbounds field[     x_idx,      y_idx]
    f₀₁ = @inbounds field[     x_idx, next_y_idx]
    f₁₀ = @inbounds field[next_x_idx,      y_idx]
    f₁₁ = @inbounds field[next_x_idx, next_y_idx]
    bottom_val = linear_interpolate(ξ, f₀₀, f₁₀)
       top_val = linear_interpolate(ξ, f₀₁, f₁₁)
    return linear_interpolate(ζ, bottom_val, top_val)
end

@inline function bicubic_interpolate(
        x_query_point, y_query_point, 
        field, field_dx, field_dy, field_dxy,
        x_nodes, y_nodes, 
        dx, N)
    x_idx, next_x_idx, ξ = get_indices(x_query_point, x_nodes, dx, N)
    y_idx, next_y_idx, ζ = get_indices(y_query_point, y_nodes, dx, N)
    
    dx2 = dx * dx
    
    f₀₀ = @inbounds field[     x_idx,      y_idx]
    f₀₁ = @inbounds field[     x_idx, next_y_idx]
    f₁₀ = @inbounds field[next_x_idx,      y_idx]
    f₁₁ = @inbounds field[next_x_idx, next_y_idx]

    fˣ₀₀ = @inbounds field_dx[     x_idx,      y_idx] * dx
    fˣ₀₁ = @inbounds field_dx[     x_idx, next_y_idx] * dx
    fˣ₁₀ = @inbounds field_dx[next_x_idx,      y_idx] * dx
    fˣ₁₁ = @inbounds field_dx[next_x_idx, next_y_idx] * dx

    fʸ₀₀ = @inbounds field_dy[     x_idx,      y_idx] * dx
    fʸ₀₁ = @inbounds field_dy[     x_idx, next_y_idx] * dx
    fʸ₁₀ = @inbounds field_dy[next_x_idx,      y_idx] * dx
    fʸ₁₁ = @inbounds field_dy[next_x_idx, next_y_idx] * dx

    fˣʸ₀₀ = @inbounds field_dxy[     x_idx,      y_idx] * dx2
    fˣʸ₀₁ = @inbounds field_dxy[     x_idx, next_y_idx] * dx2
    fˣʸ₁₀ = @inbounds field_dxy[next_x_idx,      y_idx] * dx2
    fˣʸ₁₁ = @inbounds field_dxy[next_x_idx, next_y_idx] * dx2
    
    f₀ = cubic_interpolate(ξ, f₀₀, f₁₀, fˣ₀₀, fˣ₁₀) # Interpolate along the bottom edge of the grid
    f₁ = cubic_interpolate(ξ, f₀₁, f₁₁, fˣ₀₁, fˣ₁₁) # Top edge
    
    fʸ₀ = cubic_interpolate(ξ, fʸ₀₀, fʸ₁₀, fˣʸ₀₀, fˣʸ₁₀) # Also interpolate the y-derivatives along the top and bottom
    fʸ₁ = cubic_interpolate(ξ, fʸ₀₁, fʸ₁₁, fˣʸ₀₁, fˣʸ₁₁)
         
    return cubic_interpolate(ζ, f₀, f₁, fʸ₀, fʸ₁) # Then interpolate vertically
end
end