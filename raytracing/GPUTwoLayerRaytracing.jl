using GeophysicalFlows, Printf;
using Random: seed!;
using JLD2;
using UnPack;
using LinearAlgebra: ldiv!

import .Parameters;
using .GPURaytracing;

function generate_initial_wavepackets(dev, L, k0, Npackets, sqrtNpackets)
    @devzeros typeof(dev) Float32 (Npackets, 4) wavepackets_array
    offset = L/sqrtNpackets/2;
    x = @views wavepackets_array[:, 1]
    y = @views wavepackets_array[:, 2]
    k = @views wavepackets_array[:, 3]
    l = @views wavepackets_array[:, 4]

    I = device_array(dev)(1:sqrtNpackets)
    J = device_array(dev)(1:Npackets)
    diagonal = @. I*L/sqrtNpackets - L/2 - offset
    phase = 2*pi*J/Npackets
    x .= repeat(diagonal, outer=sqrtNpackets)
    y .= repeat(diagonal, inner=sqrtNpackets)
    @. k = k0 * cos(phase)
    @. l = k0 * sin(phase)
    return wavepackets_array;
end

function savepackets!(out, clock, pos, wavenumber, velocity)
    out["p/t/$(clock.step)"] = clock.t
    out["p/x/$(clock.step)"] = Array(pos);
    out["p/k/$(clock.step)"] = Array(wavenumber);
	out["p/u/$(clock.step)"] = Array(velocity);
    return nothing;
end

function get_velocity_info(ψh, grid, params, v_info, grad_v_info, temp_in_field, temp_out_field)
    k = grid.kr;
    l = grid.l;
    
    @. temp_in_field = -1im*l*ψh;
    ldiv!(v_info.u, grid.rfftplan, temp_in_field)
    
    @. temp_in_field = 1im*k*ψh;
    ldiv!(v_info.v, grid.rfftplan, temp_in_field)
    
    @. temp_in_field = k*l*ψh;
    ldiv!(grad_v_info.ux, grid.rfftplan, temp_in_field)
    
    @. temp_in_field = l*l*ψh;
    ldiv!(grad_v_info.uy, grid.rfftplan, temp_in_field)
    
    @. temp_in_field = -k*k*ψh;
    ldiv!(grad_v_info.vx, grid.rfftplan, temp_in_field)
    
    @. grad_v_info.vy = -grad_v_info.ux
    return nothing
end

function simulate!(nsteps, nsubs, npacketsubs, grid, prob, packets, base_filename, packetSpinUpDelay, packet_params)
    # Set up memory constructs
    Dev = typeof(grid.device)

    @devzeros Dev Float32 (grid.nx, grid.ny) u1_background v1_background ux1_background uy1_background vx1_background vy1_background
    @devzeros Dev Float32 (grid.nx, grid.ny) u2_background v2_background ux2_background uy2_background vx2_background vy2_background
    
    old_velocity = Velocity(u1_background, v1_background)
    old_grad_v = VelocityGradient(ux1_background, uy1_background, vx1_background, vy1_background)
    
    new_velocity = Velocity(u2_background, v2_background)
    new_grad_v = VelocityGradient(ux2_background, uy2_background, vx2_background, vy2_background)
	
	@devzeros Dev Complex{Float32} (grid.nkr, grid.nl) temp_device_in_field
    @devzeros Dev Float32 (grid.nx, grid.ny) temp_device_out_field

    @devzeros Dev Float32 (Parameters.Npackets, 2) packet_U
    output_u = @views packet_U[:, 1]
    output_v = @views packet_U[:, 2]
    
    packet_pos = @views packets[:, 1:2]

    packet_K = @views packets[:, 3:4]
    packet_k = @views packets[:, 3]
    packet_l = @views packets[:, 4]
    
    sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

    file_index = 0
    current_writes = 1
    get_filename(file_index) = @sprintf("%s.%08d", base_filename, file_index)
    max_writes = Parameters.max_writes

    filename = get_filename(file_index)
    if isfile(filename); rm(filename); end
    out = jldopen(filename, "w")
    
    
	get_sol(prob) = prob.vars.ψh
    get_baroclinic_streamfunction(prob) = (@views(get_sol(prob)[:,:,1]) + @views(get_sol(prob)[:,:,2]))/2
    
    get_velocity_info(get_baroclinic_streamfunction(prob), grid, packet_params, old_velocity, old_grad_v, temp_device_in_field, temp_device_out_field);
    interpolate_velocity!(old_velocity, packet_pos, grid, output_u, output_v)  
    savepackets!(out, clock, packet_pos, packet_K, packet_U); # Save with latest velocity information
    
    startwalltime = time()
    frames = 0:round(Int, nsteps / npacketsubs)
	packet_frames = 1:round(Int, npacketsubs / nsubs)
    ode_template = create_template_ode(packets)

    for j=frames
        if j % (100 / nsubs) == 0
            cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])
            log = @sprintf("step: %04d, t: %.1f, cfl: %.2f, walltime: %.2f min",
                       clock.step, clock.t, cfl, (time()-startwalltime)/60)
            println(log)
            flush(stdout)
        end
        
        # Get the baroclinic velocity
		old_t = clock.t
        
		for k=packet_frames
            # Steady background flow
	        stepforward!(prob, [], nsubs);
            MultiLayerQG.updatevars!(prob);

            get_velocity_info(get_baroclinic_streamfunction(prob), grid, packet_params, new_velocity, new_grad_v, temp_device_in_field, temp_device_out_field);
            new_t = clock.t;

            raytrace!(ode_template, old_velocity, new_velocity, old_grad_v, new_grad_v, grid, packets, packet_params.dt, (old_t, new_t), packet_params);
            old_velocity = new_velocity;
            old_grad_v = new_grad_v;
            old_t = new_t;
        end
        
        mask = @. packet_k^2 + packet_l^2 >= Parameters.k_cutoff^2
        packet_k[mask] .= packet_params.k0
        packet_l[mask] .= 0

        # Update velocity info
        interpolate_velocity!(new_velocity, packet_pos, grid, output_u, output_v)
        
        savepackets!(out, clock, packet_pos, packet_K, packet_U); # Save with latest velocity information
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
    nx = size(ψh, 2)
    U = U[1,1,:]
    b = [b[1], b[2]]
    prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx=L, f₀, H, b, U, μ, β, dt, stepper, aliased_fraction=0, T=Float32)
    
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
    
    packets = generate_initial_wavepackets(device, Lx, k0, Npackets, Parameters.sqrtNpackets);
    packet_params = (f = f, Cg = Cg, dt = dt / Parameters.packetStepsPerBackgroundStep, Npackets = Npackets, k0=k0);
    simulate!(nsteps, nsubs, npacketsubs, grid, prob, packets, base_filename, Parameters.packetSpinUpDelay, packet_params);
end
