using GeophysicalFlows, Printf;
using Random: seed!;
using JLD2;
using UnPack;
using LinearAlgebra: ldiv!

import .Parameters;
import .Raytracing;

function generate_initial_wavepackets(L, k0, Npackets, sqrtNpackets)
    wavepackets = Vector{Raytracing.Wavepacket}(undef, Npackets); 
    offset = L/sqrtNpackets/2;
    for i=1:sqrtNpackets
        for j = 1:sqrtNpackets
            x = [i*L/sqrtNpackets-L/2 - offset, j*L/sqrtNpackets-L/2 - offset];
            k = k0 * [cos(2*pi*((i-1)*sqrtNpackets + j)/Npackets), sin(2*pi*((i-1)*sqrtNpackets + j)/Npackets)];
            wavepackets[(i-1)*sqrtNpackets + j] = Raytracing.Wavepacket(x, k);
        end
    end
    return wavepackets;
end

function savepackets!(out, clock, packets::AbstractVector{Raytracing.Wavepacket})
    out["p/t/$(clock.step)"] = clock.t
    for i=1:size(packets, 1)
        out["p/x/$i/$(clock.step)"] = packets[i].x;
        out["p/k/$i/$(clock.step)"] = packets[i].k;
    end    
    return nothing;
end

function get_velocity_info(ψh, grid, params, v_info, grad_v_info, temp_field)
    k = grid.kr;
    l = grid.l;
    
    @. temp_field = -params.packetVelocityScale*1im*l*ψh;
    ldiv!(v_info.u, grid.rfftplan, temp_field)
    
    @. temp_field = params.packetVelocityScale*1im*k*ψh;
    ldiv!(v_info.v, grid.rfftplan, temp_field)
    
    
    @. temp_field = params.packetVelocityScale*k*l*ψh;
    ldiv!(grad_v_info.ux, grid.rfftplan, temp_field)
    
    @. temp_field = params.packetVelocityScale*l*l*ψh;
    ldiv!(grad_v_info.uy, grid.rfftplan, temp_field)
    
    @. temp_field = -params.packetVelocityScale*k*k*ψh;
    ldiv!(grad_v_info.vx, grid.rfftplan, temp_field)
    
    @. grad_v_info.vy = -grad_v_info.ux
    return nothing
end

function get_rms_U(velocity_info::Raytracing.Velocity)
    nx, ny = size(velocity_info.u)
    return sqrt(sum(velocity_info.u.^2 + velocity_info.v.^2)/nx/ny);
end

function simulate!(nsteps, nsubs, npacketsubs, grid, prob, packets, out, packetSpinUpDelay, packet_params)
    # Set up memory constructs
    u1_background  = Array{Float64}(undef, grid.nx, grid.ny)
    v1_background  = Array{Float64}(undef, grid.nx, grid.ny)
    ux1_background = Array{Float64}(undef, grid.nx, grid.ny)
    uy1_background = Array{Float64}(undef, grid.nx, grid.ny)
    vx1_background = Array{Float64}(undef, grid.nx, grid.ny)
    vy1_background = Array{Float64}(undef, grid.nx, grid.ny)
    
    u2_background  = Array{Float64}(undef, grid.nx, grid.ny)
    v2_background  = Array{Float64}(undef, grid.nx, grid.ny)
    ux2_background = Array{Float64}(undef, grid.nx, grid.ny)
    uy2_background = Array{Float64}(undef, grid.nx, grid.ny)
    vx2_background = Array{Float64}(undef, grid.nx, grid.ny)
    vy2_background = Array{Float64}(undef, grid.nx, grid.ny)
    
    old_velocity = Raytracing.Velocity(u1_background, v1_background)
    old_grad_v = Raytracing.VelocityGradient(ux1_background, uy1_background, vx1_background, vy1_background)
    
    new_velocity = Raytracing.Velocity(u2_background, v2_background)
    new_grad_v = Raytracing.VelocityGradient(ux2_background, uy2_background, vx2_background, vy2_background)
    temp_field = Array{Complex{Float64}}(undef, grid.nkr, grid.nl)
    
    sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
    
    savepackets!(out, clock, packets)
    startwalltime = time()
    frames = 0:round(Int, nsteps / nsubs)
	packet_frames = 0:round(Int, nsubs / npacketsubs)

    for j=frames
        if j % (100 / nsubs) == 0
            cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

            log = @sprintf("step: %04d, t: %.1f, cfl: %.2f, walltime: %.2f min",
                       clock.step, clock.t, cfl, (time()-startwalltime)/60)

            println(log)
            flush(stdout)
        end
        
		get_velocity_info(@views(prob.vars.ψh[:,:,1]), grid, packet_params, old_velocity, old_grad_v, temp_field);
        old_t = clock.t
        
		for k=packet_frames
	        stepforward!(prob, [], nsubs);
            MultiLayerQG.updatevars!(prob);

            get_velocity_info(@views(prob.vars.ψh[:,:,1]), grid, packet_params, new_velocity, new_grad_v, temp_field);
            new_t = clock.t;

            Raytracing.solve!(old_velocity, new_velocity, old_grad_v, new_grad_v, grid.x, grid.y, packet_params.Npackets, packets, packet_params.dt, (old_t, new_t), packet_params);
            # stepraysforward!(grid, packets, old_v, new_v, (old_t / packet_params.packetVelocityScale, new_t / packet_params.packetVelocityScale), packet_params);
            old_velocity = new_velocity;
            old_grad_v = new_grad_v;
            old_t = new_t;
        end
        savepackets!(out, clock, packets); # Save with latest velocity information
    end 
end

function stepraysforward!(grid, packets, velocity1, dvelocity1, velocity2, dvelocity2, tspan, params)
    
end

function set_up_problem(filename, stepper)
    L = 2π
    ic_file = jldopen(filename, "r")
    ψh = ic_file["ic/ψh"]
    @unpack g, f₀, β, ρ, H, U, μ = ic_file["params"]
    dt = ic_file["clock/dt"]
    nlayers = 2
    dev = CPU();
    L = 2π
    nx = size(ψh, 2)
    U = U[1,1,:]
    ρ = [ρ[1], ρ[2]]
    prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx=L, f₀, g, H, ρ, U, μ, β, dt, stepper, aliased_fraction=0)
    pvfromstreamfunction!(prob.sol, device_array(dev)(ψh), prob.params, prob.grid)
    MultiLayerQG.updatevars!(prob)
    close(ic_file)
    return nx, dt, prob
end


function start!()
    Lx, stepper = Parameters.L, Parameters.stepper;
    nx, dt, prob = set_up_problem(Parameters.initial_condition_file, stepper);
    
    nsteps, nsubs, npacketsubs, packetSpinUpDelay = Parameters.nsteps, Parameters.nsubs, Parameters.npacketsubs, Parameters.packetSpinUpDelay
    
    sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
    f, g, H = params.f₀, params.g, params.H


    filename = joinpath(Parameters.filepath, Parameters.filename)
    if !isdir(Parameters.filepath); mkdir(Parameters.filepath); end
    if isfile(filename); rm(filename); end
    
    out = jldopen(filename, "w")

    # set_initial_condition!(dev, grid, prob, Parameters.q0_amplitude, nlayers);
    Npackets = Parameters.Npackets
    Cg = sqrt(g*H[1])
    
    # omega^2 = f^2 + gH*k^2
    # alpha^2*f^2 = f^2 + Cg^2*k^2
    # f^2(alpha^2 - 1)/gH = k^2
    # k = f/Cg*sqrt(alpha^2 - 1)
    packets = generate_initial_wavepackets(Lx, sqrt(Parameters.corFactor^2 - 1)*f/Cg, Npackets, Parameters.sqrtNpackets);
    rms_U = sqrt(sum(vars.u[:,:,1].^2 + vars.v[:,:,1].^2)/nx^2)
    packetVelocityScale = Parameters.initialFroudeNumber * Cg / rms_U
    packet_params = (f = f, Cg = Cg, dt = dt / Parameters.packetStepsPerBackgroundStep, Npackets = Npackets, packetVelocityScale = packetVelocityScale);
    simulate!(nsteps, nsubs, npacketsubs, grid, prob, packets, out, Parameters.packetSpinUpDelay, packet_params);
    close(out)
end
