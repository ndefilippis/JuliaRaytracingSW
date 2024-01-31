module Raytracing
using Interpolations;
using DifferentialEquations;
using NFFT

struct Velocity
    u::Array{Float64, 2}
    v::Array{Float64, 2}
end

struct VelocityGradient
    ux::Array{Float64, 2}
    uy::Array{Float64, 2}
    vx::Array{Float64, 2}
    vy::Array{Float64, 2}
end

function dispersion_relation(k, params)
    return sqrt(params.f^2 + params.Cg^2*(k[1]*k[1] + k[2]*k[2]));
end

function group_velocity(k, params)
   return params.Cg^2*k/dispersion_relation(k, params); 
end

function hamiltonian(x, k, params, t)
   return params.u(x[1], x[2], t)*k[1] + params.v(x[1], x[2], t) * k[2] + dispersion_relation(k, params)
end

function dxdt(xdot, x, k, params, t)
    group_vel = group_velocity(k, params);
    xdot[1] = params.u(x[1], x[2], t) + group_vel[1];
    xdot[2] = params.v(x[1], x[2], t) + group_vel[2];
end

function dkdt(kdot, x, k, params, t);
    kdot[1] = -params.ux(x[1], x[2], t)*k[1] - params.vx(x[1], x[2], t)*k[2];
    kdot[2] = -params.uy(x[1], x[2], t)*k[1] - params.vy(x[1], x[2], t)*k[2];
end

function _solve!(Npackets::Int, packets::AbstractArray{Float64, 2}, dt::Float64, tspan::Tuple{Float64, Float64}, params)
    xs = @views packets[:,1:2]
    ks = @views packets[:,3:4]
    problem = DynamicalODEProblem(dxdt, dkdt, xs, ks, tspan, params);
    sim = solve(problem, ImplicitMidpoint(), dt=dt, save_on=false, save_start=false);
    packets[i, 1] = sim[1,1]
    packets[i, 2] = sim[2,1]
    packets[i, 3] = sim[3,1]
    packets[i, 4] = sim[4,1]
    return wavepackets;
end

function solve!(psi1h, psi2h, grid,
    x::AbstractRange{Float64}, y::AbstractRange{Float64}, Npackets::Int, wavepackets::AbstractVector{Wavepacket}, dt::Float64, tspan::Tuple{Float64, Float64}, params)
    uh1 = @. -im*grid.l  * psi1h
    uh2 = @. -im*grid.l  * psi2h
    vh1 = @.  im*grid.kr * psi1h
    vh1 = @.  im*grid.kr * psi2h
    
    uxh1 = @.  grid.l*grid.kr  * psi1h
    uxh2 = @.  grid.l*grid.kr  * psi2h
    uyh1 = @.  grid.l*grid.l   * psi1h
    uyh2 = @.  grid.l*grid.l   * psi2h
    vxh1 = @. -grid.kr*grid.kr * psi1h
    vxh1 = @. -grid.kr*grid.kr * psi2h
    vyh1 = @. -grid.kr*grid.l  * psi1h
    vyh1 = @. -grid.kr*grid.l  * psi2h
    
    velocity_params = (u = (x, y, t) ->  interp_fields(x, y, uh1, uh2, tspan, t),
                       v = (x, y, t) ->  interp_fields(x, y, vh1, vh2, tspan, t),
                       ux = (x, y, t) -> interp_fields(x, y, uxh1, uxh2, tspan, t),
                       uy = (x, y, t) -> interp_fields(x, y, uyh1, uyh2, tspan, t),
                       vx = (x, y, t) -> interp_fields(x, y, vxh1, vxh2, tspan, t),
                       vy = (x, y, t) -> interp_fields(x, y, vyh1, vyh2, tspan, t),
    params = merge(params, velocity_params);
    return _solve!(Npackets, wavepackets, dt, tspan, params);
end

function interp_fields(x, y, f1h, f2h, tspan, t)
    nx, ny = size(f1h)
    p1 = nufft2d2(x, y, 1, 1e-5, f1h, modeord=1) 
    p2 = nufft2d2(x, y, 1, 1e-5, f2h, modeord=1)
    alpha = (t - tspan[1])/(tspan[2] - tspan[1])
    return (alpha * p1 + (1 - alpha)*p2) / (nx * ny)
end
end
