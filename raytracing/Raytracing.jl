module Raytracing
using Interpolations;
using OrdinaryDiffEq;

struct Wavepacket
    x::Vector{Float64}
    k::Vector{Float64}
	u::Vector{Float64}
end

struct Velocity
    u::AbstractArray{Float64, 2}
    v::AbstractArray{Float64, 2}
end

struct VelocityInterpolator
    u::AbstractInterpolation
    v::AbstractInterpolation
end

struct VelocityGradient
    ux::AbstractArray{Float64, 2}
    uy::AbstractArray{Float64, 2}
    vx::AbstractArray{Float64, 2}
    vy::AbstractArray{Float64, 2}
end

struct VelocityGradientInterpolator
    ux::AbstractInterpolation
    uy::AbstractInterpolation
    vx::AbstractInterpolation
    vy::AbstractInterpolation
end

function createVelocityInterpolator(U::Velocity, x::AbstractRange{Float64}, y::AbstractRange{Float64})
    u_itp = interpolator(U.u, x, y);
    v_itp = interpolator(U.v, x, y);
    return VelocityInterpolator(u_itp, v_itp);
end

function createVelocityInterpolator(U1::Velocity, U2::Velocity, x::AbstractRange{Float64}, y::AbstractRange{Float64}, tspan)
    u_itp = interpolator(U1.u, U2.u, x, y, tspan);
    v_itp = interpolator(U1.v, U2.v, x, y, tspan);
    return VelocityInterpolator(u_itp, v_itp);
end

function createVelocityGradientInterpolator(dU::VelocityGradient, x::AbstractRange{Float64}, y::AbstractRange{Float64})
    ux_itp = interpolator(dU.ux, x, y);
    uy_itp = interpolator(dU.uy, x, y);
    vx_itp = interpolator(dU.vx, x, y);
    vy_itp = interpolator(dU.vy, x, y);
    return VelocityGradientInterpolator(ux_itp, uy_itp, vx_itp, vy_itp);
end

function createVelocityGradientInterpolator(dU1::VelocityGradient, dU2::VelocityGradient, x::AbstractRange{Float64}, y::AbstractRange{Float64}, tspan)
    ux_itp = interpolator(dU1.ux, dU2.ux, x, y, tspan);
    uy_itp = interpolator(dU1.uy, dU2.uy, x, y, tspan);
    vx_itp = interpolator(dU1.vx, dU2.vx, x, y, tspan);
    vy_itp = interpolator(dU1.vy, dU2.vy, x, y, tspan);
    return VelocityGradientInterpolator(ux_itp, uy_itp, vx_itp, vy_itp);
end

@nospecialize
function dispersion_relation(k, params)
    return sqrt(params.f^2 + params.Cg^2*(k[1]*k[1] + k[2]*k[2]));
end

@nospecialize
function group_velocity(k, params)
   return params.Cg^2*k/dispersion_relation(k, params); 
end

@nospecialize
function hamiltonian(x, k, params, t)
   return params.u(x[1], x[2], t)*k[1] + params.v(x[1], x[2], t) * k[2] + dispersion_relation(k, params)
end

@nospecialize
function dxdt(xdot, x, k, params, t)
    group_vel = group_velocity(k, params);
    xdot[1] = params.u(x[1], x[2], t) + group_vel[1];
    xdot[2] = params.v(x[1], x[2], t) + group_vel[2];
end

@nospecialize
function dkdt(kdot, x, k, params, t);
    kdot[1] = -params.ux(x[1], x[2], t)*k[1] - params.vx(x[1], x[2], t)*k[2];
    kdot[2] = -params.uy(x[1], x[2], t)*k[1] - params.vy(x[1], x[2], t)*k[2];
end

function _solve!(Npackets::Int, wavepackets::AbstractVector{Wavepacket}, dt::Float64, tspan::Tuple{Float64, Float64}, params)
    #ics = Vector(undef, Npackets);
    #for i=1:Npackets
    #    ics[i] = ArrayPartition(wavepackets[i].x, wavepackets[i].k);
    #end
    #function prob_func(prob, i, repeat)
    #    remake(prob, u0 = ics[i]);
    #end
	#output_func(sol, i) = (sol[end], false)
    #println("Create problem:")
    #@time problem = DynamicalODEProblem(dxdt, dkdt, ics[1][1:2], ics[1][3:4], tspan, params);
    #println("Create ensemble:")
    #@time ensemble_prob = EnsembleProblem(problem, prob_func = prob_func, output_func = output_func, safetycopy=false);
    #println("Solve:")
    #@time sim = solve(ensemble_prob, ImplicitMidpoint(), EnsembleThreads(), trajectories=Npackets, dt=dt, save_on=false, save_start=false)
    Threads.@threads for i=1:Npackets
	#for i=1:Npackets
        problem = DynamicalODEProblem(dxdt, dkdt, wavepackets[i].x, wavepackets[i].k, tspan, params);
        local sim = solve(problem, ImplicitMidpoint(), dt=dt, save_on=false, save_start=false);
        wavepackets[i].x[1] = sim[1,1];
		wavepackets[i].x[2] = sim[2,1];
		wavepackets[i].k[1] = sim[3,1];
		wavepackets[i].k[2] = sim[4,1];
		wavepackets[i].u[1] = params.u(wavepackets[i].x[1], wavepackets[i].x[2], tspan[2])
		wavepackets[i].u[2] = params.v(wavepackets[i].x[1], wavepackets[i].x[2], tspan[2])
    end
    return wavepackets;
end

function solve!(velocity::Velocity, gradient::VelocityGradient, x::AbstractRange{Float64}, y::AbstractRange{Float64}, Npackets::Int, wavepackets::AbstractVector{Wavepacket}, dt::Float64, tspan::Tuple{Float64, Float64}, params)

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
    x::AbstractRange{Float64}, y::AbstractRange{Float64}, Npackets::Int, wavepackets::AbstractVector{Wavepacket}, dt::Float64, tspan::Tuple{Float64, Float64}, params)

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
                interpolate(field, BSpline(Cubic(Periodic(OnCell())))), 
            x, y), 
        Periodic(OnCell()));
end

function interpolator(field1::Array{Float64, 2}, field2::Array{Float64, 2}, x::AbstractRange{Float64}, y::AbstractRange{Float64}, tspan)
    # return cubic_spline_interpolation((grid.x, grid.y), field, extrapolation_bc = Periodic());
    field = cat(field1, field2, dims=3);
    return extrapolate(
            scale(
                interpolate(field, 
                    (BSpline(Quadratic(Periodic(OnGrid()))), BSpline(Quadratic(Periodic(OnGrid()))), BSpline(Linear()))),
            x, y, range(tspan[1], tspan[2], length=2)), 
          (Periodic(), Periodic(), Linear()));
end
end
