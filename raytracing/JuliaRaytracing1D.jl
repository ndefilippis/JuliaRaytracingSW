function ω(k, p)
    return sqrt.(1 + k[1]*k[1] + k[2]*k[2])
end

function dxkdt(dxk, xk, p, t)
    xs = @views xk[1:p.Np]
    ys = @views xk[(p.Np+1):(2*p.Np)]
    nx = @. (xs - p.x0) / p.Lx + 0.5 / p.N
    ny = @. (ys - p.x0) / p.Lx + 0.5 / p.N
    
    k1 = @views xk[(2*p.Np+1):(3*p.Np)]
    k2 = @views xk[(3*p.Np+1):(4*p.Np)]

    ω = @. sqrt(1 + k1^2 + k2^2)
    Cg_x = @. k1 / ω
    Cg_y = @. k2 / ω

    dx1 = @views dxk[(0*p.Np+1):(1*p.Np)]
    dx2 = @views dxk[(1*p.Np+1):(2*p.Np)]
    dk1 = @views dxk[(2*p.Np+1):(3*p.Np)]
    dk2 = @views dxk[(3*p.Np+1):(4*p.Np)]
    
    broadcast!(dx1, nx, ny, Cg_x, Ref(p.U)) do xi, yi, cg, U
        U[xi, yi] + cg
    end
    broadcast!(dx2, nx, ny, Cg_y, Ref(p.V)) do xi, yi, cg, V
        V[xi, yi] + cg
    end
    
    broadcast!(dk1, nx, ny, k1, k2, Ref(p.Ux), Ref(p.Vx)) do xi, yi, k, l, Ux, Vx
        -(Ux[xi, yi] * k + Vx[xi, yi] * l)
    end
    broadcast!(dk2, nx, ny, k1, k2, Ref(p.Uy), Ref(p.Ux)) do xi, yi, k, l, Uy, negVy
        -(Uy[xi, yi] * k - negVy[xi, yi] * l)
    end
end

function dxdt(dx, k, x, p, t)
    xs = @views x[(0*p.Np+1):(1*p.Np)]
    ys = @views x[(p.Np+1):(2*p.Np)]
    nx = @. (xs - p.x0) / p.Lx + 0.5 / p.N
    ny = @. (ys - p.x0) / p.Lx + 0.5 / p.N
    
    k1 = @views k[1:p.Np]
    k2 = @views k[(p.Np+1):(2*p.Np)]

    ω = @. sqrt(1 + k1^2 + k2^2)
    Cg_x = @. k1 / ω
    Cg_y = @. k2 / ω

    dx1 = @views dx[(0*p.Np+1):(1*p.Np)]
    dx2 = @views dx[(1*p.Np+1):(2*p.Np)]
    
    broadcast!(dx1, nx, ny, Cg_x, Ref(p.U)) do xi, yi, cg, U
        U[xi, yi] + cg
    end
    broadcast!(dx2, nx, ny, Cg_y, Ref(p.V)) do xi, yi, cg, V
        V[xi, yi] + cg
    end
end

function dkdt(dk, x, k, p, t)
    xs = @views x[(0*p.Np+1):(1*p.Np)]
    ys = @views x[(p.Np+1):(2*p.Np)]
    nx = @. (xs - p.x0) / p.Lx + 0.5 / p.N
    ny = @. (ys - p.x0) / p.Lx + 0.5 / p.N
    
    k1 = @views k[(0*p.Np+1):(1*p.Np)]
    k2 = @views k[(1*p.Np+1):(2*p.Np)]

    dk1 = @views dk[(0*p.Np+1):(1*p.Np)]
    dk2 = @views dk[(1*p.Np+1):(2*p.Np)]
    
    broadcast!(dk1, nx, ny, k1, k2, Ref(p.Ux), Ref(p.Vx)) do xi, yi, k, l, Ux, Vx
        -(Ux[xi, yi] * k + Vx[xi, yi] * l)
    end
    broadcast!(dk2, nx, ny, k1, k2, Ref(p.Uy), Ref(p.Ux)) do xi, yi, k, l, Uy, negVy
        -(Uy[xi, yi] * k - negVy[xi, yi] * l)
    end
end

Np = 150
xk0 = CuArray{Float32}(undef, (4 * Np))

x0  = CuArray{Float32}(undef, (2 * Np))
k0 = CuArray{Float32}(undef, (2 * Np))

x0 .= cu(grid.Lx * rand(2*Np) .- grid.Lx/2)
@views xk0[1:2*Np] .= x0

phase = 2π*CUDA.rand(Float32, Np)

@views k0[(0*Np+1):1*Np] .= cos.(phase)
@views k0[(1*Np+1):2*Np] .= sin.(phase)

@views xk0[2*Np+1:end] .= k0

texU = CuTexture(U; interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
texV = CuTexture(V; interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
texUx = CuTexture(Ux; interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
texUy = CuTexture(Uy; interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)
texVx = CuTexture(Vx; interpolation=CUDA.LinearInterpolation(), address_mode=CUDA.ADDRESS_MODE_WRAP, normalized_coordinates=true)

color = repeat(I(Np), 1, 4)
#color = zeros(Bool, 15, 4, 15)
params = (U = texU, V = texV, Ux = texUx, Uy = texUy, Vx = texVx, x0 = grid.x[1], Lx=grid.Lx, N = grid.nx, Np = Np)
f = ODEFunction(dxkdt)#, jac_prototype=color)
prob = ODEProblem(f, xk0, (0.0f0, 100.0f0), params) # Float32 is better on GPUs!

sol1 = @btime solve(prob, Vern7(), save_start=true, save_on=false)
sol2 = @btime solve(prob, Vern8(), save_start=true, save_on=false)