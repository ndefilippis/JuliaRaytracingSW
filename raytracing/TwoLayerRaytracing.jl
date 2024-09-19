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
            wavepackets[(i-1)*sqrtNpackets + j] = Raytracing.Wavepacket(x, k, [0, 0]);
        end
    end
    return wavepackets;
end

function savepackets!(out, clock, packets::AbstractVector{Raytracing.Wavepacket})
    out["p/t/$(clock.step)"] = clock.t
    for i=1:size(packets, 1)
        out["p/x/$i/$(clock.step)"] = packets[i].x;
        out["p/k/$i/$(clock.step)"] = packets[i].k;
		out["p/u/$i/$(clock.step)"] = packets[i].u
    end    
    return nothing;
end

function get_velocity_info(ψh, grid, params, v_info, grad_v_info, temp_in_field, temp_out_field)
    k = grid.kr;
    l = grid.l;
    
    @. temp_in_field = -1im*l*ψh;
    ldiv!(temp_out_field, grid.rfftplan, temp_in_field)
	@. v_info.u = temp_out_field
    
    @. temp_in_field = 1im*k*ψh;
    ldiv!(temp_out_field, grid.rfftplan, temp_in_field)
    @. v_info.v = temp_out_field
    
    
    @. temp_in_field = k*l*ψh;
    ldiv!(temp_out_field, grid.rfftplan, temp_in_field)
    @. grad_v_info.ux = temp_out_field
    
    @. temp_in_field = l*l*ψh;
    ldiv!(temp_out_field, grid.rfftplan, temp_in_field)
    @. grad_v_info.uy = temp_out_field
    
    @. temp_in_field = -k*k*ψh;
    ldiv!(temp_out_field, grid.rfftplan, temp_in_field)
    @. grad_v_info.vx = temp_out_field
    
    @. grad_v_info.vy = -grad_v_info.ux
    return nothing
end

function get_rms_U(velocity_info::Raytracing.Velocity)
    nx, ny = size(velocity_info.u)
    return sqrt(sum(velocity_info.u.^2 + velocity_info.v.^2)/nx/ny);
end

function simulate!(nsteps, nsubs, npacketsubs, grid, prob, packets, base_filename, packetSpinUpDelay, packet_params)
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
	
	Dev = typeof(grid.device)
	@devzeros Dev Complex{Float64} (grid.nkr, grid.nl) temp_device_in_field
    @devzeros Dev Float64 (grid.nx, grid.ny) temp_device_out_field
    
    sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

    file_index = 0
    current_writes = 1
    get_filename(file_index) = @sprintf("%s.%08d", base_filename, file_index)
    max_writes = Parameters.max_writes

    filename = get_filename(file_index)
    if isfile(filename); rm(filename); end
    out = jldopen(filename, "w")
    
    savepackets!(out, clock, packets)
    startwalltime = time()
    frames = 0:round(Int, nsteps / npacketsubs)
	packet_frames = 1:round(Int, npacketsubs / nsubs)

	get_sol(prob) = prob.vars.ψh

    for j=frames
        if j % (100 / nsubs) == 0
            cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

            log = @sprintf("step: %04d, t: %.1f, cfl: %.2f, walltime: %.2f min",
                       clock.step, clock.t, cfl, (time()-startwalltime)/60)

            println(log)
            flush(stdout)
        end

		get_background_u(prob) = (@views(get_sol(prob)[:,:,1]) + @views(get_sol(prob)[:,:,2]))/2
        
		get_velocity_info(get_background_u(prob), grid, packet_params, old_velocity, old_grad_v, temp_device_in_field, temp_device_out_field);
        old_t = clock.t
        
		for k=packet_frames
            # Steady background flow
	        stepforward!(prob, [], nsubs);
            MultiLayerQG.updatevars!(prob);

             get_velocity_info(get_background_u(prob), grid, packet_params, new_velocity, new_grad_v, temp_device_in_field, temp_device_out_field);
            new_t = clock.t;

            Raytracing.solve!(old_velocity, new_velocity, old_grad_v, new_grad_v, grid.x, grid.y, packet_params.Npackets, packets, packet_params.dt, (packet_params.packetVelocityScale * old_t, packet_params.packetVelocityScale * new_t), packet_params);
            for i=1:packet_params.Npackets
                if(packets[i].k[1]^2 + packets[i].k[2]^2 >= Parameters.k_cutoff^2)
                    packets[i].k[1] = packet_params.k0
                    packets[i].k[2] = 0
                end
            end
            # stepraysforward!(grid, packets, old_v, new_v, (old_t / packet_params.packetVelocityScale, new_t / packet_params.packetVelocityScale), packet_params);
            old_velocity = new_velocity;
            old_grad_v = new_grad_v;
            old_t = new_t;
            clock.step += 1
        end
        clock.step += 1
        savepackets!(out, clock, packets); # Save with latest velocity information
        current_writes += 1
        if current_writes >= max_writes
            close(out)
            current_writes = 0
            file_index += 1
            filename = get_filename(file_index)
            out = jldopen(filename, "w")
        end
    end 
    close(out)
end

function set_up_problem(filename, stepper, dev)
    L = 2π
    ic_file = jldopen(filename, "r")
	index = keys(ic_file["snapshots/t"])[1]
    ψh = ic_file["snapshots/ψh/$index"]
    @unpack f₀, β, b, H, U, μ = ic_file["params"]
    dt = ic_file["clock/dt"]
    nlayers = 2
    L = 2π
    nx = size(ψh, 2)
    U = U[1,1,:]
    b = [b[1], b[2]]
    prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx=L, f₀, H, b, U, μ, β, dt, stepper, aliased_fraction=0)
    
	#MultiLayerQG.streamfunctionfrompv!(prob.vars.ψh, device_array(dev)(ψh), prob.params, prob.grid)
	#MultiLayerQG.pvfromstreamfunction!(prob.sol, prob.vars.ψh, prob.params, prob.grid)
	MultiLayerQG.pvfromstreamfunction!(prob.sol, device_array(dev)(ψh), prob.params, prob.grid)
    MultiLayerQG.updatevars!(prob)
    close(ic_file)
    return nx, dt, prob
end


function start!()
    Lx, stepper, device = Parameters.L, Parameters.stepper, Parameters.device;
    nx, dt, prob = set_up_problem(Parameters.initial_condition_file, stepper, device);
    
    total_time, nsubs, npacketsubs, packetSpinUpDelay = Parameters.total_time, Parameters.nsubs, Parameters.npacketsubs, Parameters.packetSpinUpDelay
    
	nsteps = Int(ceil(total_time / dt))
	println("Number of steps: ", nsteps)
    sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
    f = params.f₀
	g = 1
	H = 1


    
    base_filename = joinpath(Parameters.filepath, Parameters.filename)
    if !isdir(Parameters.filepath); mkdir(Parameters.filepath); end
    #if isfile(filename); rm(filename); end
    
    #out = jldopen(filename, "w")

    # set_initial_condition!(dev, grid, prob, Parameters.q0_amplitude, nlayers);
    Npackets = Parameters.Npackets
    Cg = sqrt(g*H)
    
    # omega^2 = f^2 + gH*k^2
    # alpha^2*f^2 = f^2 + Cg^2*k^2
    # f^2(alpha^2 - 1)/gH = k^2
    # k = f/Cg*sqrt(alpha^2 - 1)

    k0 = sqrt(Parameters.corFactor^2 - 1)*f/Cg
    packets = generate_initial_wavepackets(Lx, k0, Npackets, Parameters.sqrtNpackets);
    rms_U = sqrt(sum(vars.u[:,:,1].^2 + vars.v[:,:,1].^2)/nx^2)
    packetVelocityScale = 1 # Parameters.initialFroudeNumber * Cg / rms_U
    packet_params = (f = f, Cg = Cg / packetVelocityScale, dt = dt / Parameters.packetStepsPerBackgroundStep, Npackets = Npackets, packetVelocityScale = packetVelocityScale, k0=k0);
    simulate!(nsteps, nsubs, npacketsubs, grid, prob, packets, base_filename, Parameters.packetSpinUpDelay, packet_params);
end
