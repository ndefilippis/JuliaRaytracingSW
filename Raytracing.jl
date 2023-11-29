module Raytracing
using Interpolations;
using GeometricIntegrators;

struct Wavepacket
    x::Vector{Float64}
    k::Vector{Float64}
end

struct Velocity
    u::Array{Float64, 2}
    v::Array{Float64, 2}
end

struct VelocityInterpolator
    u::AbstractInterpolation
    v::AbstractInterpolation
end

struct VelocityGradient
    ux::Array{Float64, 2}
    uy::Array{Float64, 2}
    vx::Array{Float64, 2}
    vy::Array{Float64, 2}
end

struct VelocityGradientInterpolator
    ux::AbstractInterpolation
    uy::AbstractInterpolation
    vx::AbstractInterpolation
    vy::AbstractInterpolation
end

function createVelocityInterpolator(U::Velocity, x, y)
    u_itp = interpolator(U.u, x, y);
    v_itp = interpolator(U.v, x, y);
    return VelocityInterpolator(u_itp, v_itp);
end

function createVelocityInterpolator(U1::Velocity, U2::Velocity, x, y, tspan)
    u_itp = interpolator(U1.u, U2.u, x, y, tspan);
    v_itp = interpolator(U1.v, U2.v, x, y, tspan);
    return VelocityInterpolator(u_itp, v_itp);
end

function createVelocityGradientInterpolator(dU::VelocityGradient, x, y)
    ux_itp = interpolator(dU.ux, x, y);
    uy_itp = interpolator(dU.uy, x, y);
    vx_itp = interpolator(dU.vx, x, y);
    vy_itp = interpolator(dU.vy, x, y);
    return VelocityGradientInterpolator(ux_itp, uy_itp, vx_itp, vy_itp);
end

function createVelocityGradientInterpolator(dU1::VelocityGradient, dU2::VelocityGradient, x, y, tspan)
    ux_itp = interpolator(dU1.ux, dU2.ux, x, y, tspan);
    uy_itp = interpolator(dU1.uy, dU2.uy, x, y, tspan);
    vx_itp = interpolator(dU1.vx, dU2.vx, x, y, tspan);
    vy_itp = interpolator(dU1.vy, dU2.vy, x, y, tspan);
    return VelocityGradientInterpolator(ux_itp, uy_itp, vx_itp, vy_itp);
end

function dispersion_relation(k, params)
    return sqrt(params.f^2 + params.Cg^2*(k[1]*k[1] + k[2]*k[2]));
end

function group_velocity(k, params)
   return params.Cg^2*k/dispersion_relation(k, params); 
end

function hamiltonian(t, x, k, params)
   return params.u(x[1], x[2], t)*k[1] + params.v(x[1], x[2], t) * k[2] + dispersion_relation(k, params)
end


function dxdt(xdot, t, x, k, params)
    group_vel = group_velocity(k, params);
    xdot[1] = params.u(x[1], x[2], t) + group_vel[1];
    xdot[2] = params.v(x[1], x[2], t) + group_vel[2];
end

function dkdt(kdot, t, x, k, params);
    kdot[1] = -params.ux(x[1], x[2], t)*k[1] - params.vx(x[1], x[2], t)*k[2];
    kdot[2] = -params.uy(x[1], x[2], t)*k[1] - params.vy(x[1], x[2], t)*k[2];
end

function _solve!(Npackets::Int, wavepackets::AbstractVector{Wavepacket}, dt::Float64, tspan::Tuple{Float64, Float64}, params)
    ics = Array{NamedTuple{(:q, :p), Tuple{Vector{Float64}, Vector{Float64}}}}(undef, Npackets);
    Threads.@threads for i=1:Npackets
        ic = (q = wavepackets[i].x, p = wavepackets[i].k);
        problem = PODEProblem(dxdt, dkdt, tspan, dt, ic, parameters=params);
        integrator = GeometricIntegrator(problem, ImplicitMidpoint());
        sol = integrate(integrator);
        wavepackets[i] = Wavepacket(sol.q[end], sol.p[end]);
    end
    return solutions;
end

function solve!(velocity::Velocity, gradient::VelocityGradient, x, y, Npackets::Int, wavepackets::AbstractVector{Wavepacket}, dt::Float64, tspan::Tuple{Float64, Float64}, params)

    velocityInterpolator = createVelocityInterpolator(velocity, x, y);
    velocityGradientInterpolator = createVelocityGradientInterpolator(gradient, x, y);

    velocity_params = (u = (x, y, t) -> velocityInterpolator.u(x, y),
                       v = (x, y, t) -> velocityInterpolator.v(x, y),
                       ux = (x, y, t) -> velocityGradientInterpolator.ux(x, y),
                       uy = (x, y, t) -> velocityGradientInterpolator.uy(x, y),
                       vx = (x, y, t) -> velocityGradientInterpolator.vx(x, y),
                       vy = (x, y, t) -> velocityGradientInterpolator.vy(x, y));

    params = merge(params, velocity_params);
    return _solve!(Npackets, wavepackets, dt, tspan, params);
end

function solve!(velocity1::Velocity, velocity2::Velocity, gradient1::VelocityGradient, gradient2::VelocityGradient, 
    x, y, Npackets::Int, wavepackets::AbstractVector{Wavepacket}, dt::Float64, tspan::Tuple{Float64, Float64}, params)

    velocityInterpolator = createVelocityInterpolator(velocity1, velocity2, x, y, tspan);
    velocityGradientInterpolator = createVelocityGradientInterpolator(gradient1, gradient2, x, y, tspan);

    velocity_params = (u = (x, y, t) -> velocityInterpolator.u(x, y, t),
                       v = (x, y, t) -> velocityInterpolator.v(x, y, t),
                       ux = (x, y, t) -> velocityGradientInterpolator.ux(x, y, t),
                       uy = (x, y, t) -> velocityGradientInterpolator.uy(x, y, t),
                       vx = (x, y, t) -> velocityGradientInterpolator.vx(x, y, t),
                       vy = (x, y, t) -> velocityGradientInterpolator.vy(x, y, t));
    params = merge(params, velocity_params);
    return _solve!(Npackets, wavepackets, dt, tspan, params);
end

function interpolator(field, x, y)
    # return cubic_spline_interpolation((grid.x, grid.y), field, extrapolation_bc = Periodic());
    return extrapolate(
            scale(
                interpolate(field, BSpline(Quadratic(Periodic(OnCell())))), 
            x, y), 
        Periodic());
end

function interpolator(field1, field2, x, y, tspan)
    # return cubic_spline_interpolation((grid.x, grid.y), field, extrapolation_bc = Periodic());
    field = cat(field1, field2, dims=3);
    return extrapolate(
            scale(
                interpolate(field, 
                    (BSpline(Quadratic(Periodic(OnCell()))), BSpline(Quadratic(Periodic(OnCell()))), BSpline(Linear()))),
            x, y, range(tspan[1], tspan[2], length=2)), 
          (Periodic(), Periodic(), Linear()));
end
end
