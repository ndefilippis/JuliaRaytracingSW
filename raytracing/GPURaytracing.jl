module GPURaytracing
using CUDA, OrdinaryDiffEq

export Velocity, VelocityGradient, raytrace!, interpolate_velocity!, interpolate_gradients!, create_template_ode

struct Velocity{T<:AbstractFloat}
    u::CuArray{T, 2}
    v::CuArray{T, 2}
end

struct VelocityGradient{T<:AbstractFloat}
    ux::CuArray{T, 2}
    uy::CuArray{T, 2}
    vx::CuArray{T, 2}
    vy::CuArray{T, 2}
end

@inline function normalize_to_texture_coords(x, x0, L, nx)
    return (x - x0) / L + 0.5 / nx
end

@inline function dispersion_relation(k, l, f, Cg)
    return sqrt(f^2 + Cg^2*(k^2 + l^2))
end


@inline function dispersion_relation(k, l, f, Cg, pos_neg)
    return pos_neg * sqrt(f^2 + Cg^2*(k^2 + l^2))
end

# Store dk as (N, 4), its fast to extract the x-component since its in a column
function dxkdt(dxk, xk, p, t)
    alpha = (t - p.t0) / (p.t1 - p.t0)

    x = @views xk[:, 1:2]
    norm_x = @. normalize_to_texture_coords(x, p.x0, p.Lx, p.Nx)
    nx = @views norm_x[:, 1]
    ny = @views norm_x[:, 2]

    k1 = @views xk[:, 3]
    k2 = @views xk[:, 4]

    ω = @. dispersion_relation(k1, k2, p.f, p.Cg, p.pos_neg)
    Cg_x = @. p.Cg^2 * k1 / ω
    Cg_y = @. p.Cg^2 * k2 / ω

    dx1 = @views dxk[:, 1]
    dx2 = @views dxk[:, 2]
    dk1 = @views dxk[:, 3]
    dk2 = @views dxk[:, 4]
    
    broadcast!(dx1, nx, ny, Cg_x, Ref(p.U1), Ref(p.U2), alpha) do xi, yi, cgx, U1, U2, alpha
        alpha * U1[xi, yi] + (1.0 - alpha) * U2[xi, yi] + cgx
    end
    broadcast!(dx2, nx, ny, Cg_y, Ref(p.V1), Ref(p.V2), alpha) do xi, yi, cgy, V1, V2, alpha
        alpha * V1[xi, yi] + (1.0 - alpha) * V2[xi, yi] + cgy
    end
    
    broadcast!(dk1, nx, ny, k1, k2, Ref(p.Ux1), Ref(p.Vx1), Ref(p.Ux2), Ref(p.Vx2), alpha) do xi, yi, k, l, Ux1, Vx1, Ux2, Vx2, alpha
        -alpha * (Ux1[xi, yi] * k + Vx1[xi, yi] * l) + (alpha - 1.0) * (Ux2[xi, yi] * k + Vx2[xi, yi] * l)
    end
    broadcast!(dk2, nx, ny, k1, k2, Ref(p.Uy1), Ref(p.Ux1), Ref(p.Uy2), Ref(p.Ux2), alpha) do xi, yi, k, l, Uy1, negVy1, Uy2, negVy2, alpha
        -alpha * (Uy1[xi, yi] * k - negVy1[xi, yi] * l) + (alpha - 1.0) * (Uy2[xi, yi] * k - negVy2[xi, yi] * l)
    end
end

function interpolate_velocity!(velocity::Velocity, positions, grid, output_U, output_V)
    texU  = CuTexture(velocity.u;  interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
    texV  = CuTexture(velocity.v;  interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)

    norm_coords = @. normalize_to_texture_coords(positions, grid.x[1], grid.Lx, grid.nx)
    nx = @views norm_coords[:, 1]
    ny = @views norm_coords[:, 2]
    
    broadcast!(output_U, nx, ny, Ref(texU)) do xi, yi, U
        U[xi, yi]
    end

    broadcast!(output_V, nx, ny, Ref(texV)) do xi, yi, V
        V[xi, yi]
    end
end

function interpolate_gradients!(gradient::VelocityGradient, positions, grid, output_Ux, output_Uy, output_Vx, output_Vy)
    texUx  = CuTexture(gradient.ux;  interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
    texUy  = CuTexture(gradient.uy;  interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
    texVx  = CuTexture(gradient.vx;  interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
    texVy  = CuTexture(gradient.vy;  interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)

    norm_coords = @. normalize_to_texture_coords(positions, grid.x[1], grid.Lx, grid.nx)
    nx = @views norm_coords[:, 1]
    ny = @views norm_coords[:, 2]
    
    broadcast!(output_Ux, nx, ny, Ref(texUx)) do xi, yi, Ux
        Ux[xi, yi]
    end

    broadcast!(output_Uy, nx, ny, Ref(texUy)) do xi, yi, Uy
        Uy[xi, yi]
    end

    broadcast!(output_Vx, nx, ny, Ref(texVx)) do xi, yi, Vx
        Vx[xi, yi]
    end

    broadcast!(output_Vy, nx, ny, Ref(texVy)) do xi, yi, Vy
        Vy[xi, yi]
    end
end

function create_template_ode(wavepacket_array)
    return ODEProblem(dxkdt, wavepacket_array, (0.0, 1.0), (0.0, )) # Place-holder ODE template
end

function raytrace!(ode_template, velocity1::Velocity, velocity2::Velocity, gradient1::VelocityGradient, gradient2::VelocityGradient, 
        grid, wavepacket_array, dt, tspan::Tuple{T, T}, params) where {T<:AbstractFloat}
    
    texU1  = CuTexture(velocity1.u;  interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
    texV1  = CuTexture(velocity1.v;  interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
    texUx1 = CuTexture(gradient1.ux; interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
    texUy1 = CuTexture(gradient1.uy; interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
    texVx1 = CuTexture(gradient1.vx; interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
    texU2  = CuTexture(velocity2.u;  interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
    texV2  = CuTexture(velocity2.v;  interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
    texUx2 = CuTexture(gradient2.ux; interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
    texUy2 = CuTexture(gradient2.uy; interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
    texVx2 = CuTexture(gradient2.vx; interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
    
    ode_params = (
        U1 = texU1, V1 = texV1, Ux1 = texUx1, Uy1 = texUy1, Vx1 = texVx1, 
        U2 = texU2, V2 = texV2, Ux2 = texUx2, Uy2 = texUy2, Vx2 = texVx2, 
        x0 = grid.x[1], Lx=grid.Lx, Nx = grid.nx, f=params.f, Cg=params.Cg, 
        pos_neg = params.frequency_sign,
        t0 = tspan[1], t1 = tspan[2])
    prob = remake(ode_template; u0=wavepacket_array, tspan=tspan, p=ode_params)

    sol = solve(prob, Vern7(), save_start=false, save_on=false, force_dtmin=true)

    # Copy solution to the original destination
    wavepacket_array .= first(sol.u)
    return nothing
end
end
