using Printf;
using FourierFlows
using Random: seed!;
using JLD2;
using LinearAlgebra: ldiv!

import .Parameters;
using .GPURaytracing;
using .SWQG
using .SequencedOutputs
using .GPURaytracing;

function set_shafer_initial_condition_QG!(prob, Kg, ag)
    grid = prob.grid
    dev = typeof(grid.device)
    T = typeof(grid.Lx)

    @devzeros dev Complex{T} (grid.nkr, grid.nl) ψh
    @devzeros dev T (grid.nx, grid.ny) ψ u
    
    geo_filter  = Kg[1]^2 .<= grid.Krsq .<= Kg[2]^2
    phase = device_array(grid.device)(2π*rand(grid.nkr, grid.nl))
    shift = exp.(1im * phase)
    ψh[geo_filter] += 0.5*shift[geo_filter]

    ldiv!(ψ, grid.rfftplan, deepcopy(ψh))
    ldiv!(u, grid.rfftplan, -grid.l .* ψh)
    
    ψh *= ag / maximum(abs.(u))
    SWQG.set_solution!(prob, ψh)
end

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

function initialize_problem()
    dev = GPU()
    Lx=Parameters.L
    nx=Parameters.nx
    dx=Lx/nx
    kmax = nx/2 - 1
    nν=Parameters.nν
    νtune = Parameters.νtune
    
    cfltune = Parameters.cfltune
    umax = Parameters.ag
    
    dt = cfltune / umax * dx
    ν = νtune * 2π / nx / (kmax^(2*nν)) / dt
    
    nsteps = ceil(Int, Parameters.T / dt)
    spinup_step = floor(Int, Parameters.spinup_T / dt)
    packet_output_freq = max(floor(Int, Parameters.packet_output_dt / dt), 1)
    output_per_packet_freq = max(floor(Int, Parameters.output_dt / Parameters.packet_output_dt), 1)
    output_freq = output_per_packet_freq * packet_output_freq
    diags_freq = max(floor(Int, Parameters.diag_dt / dt), 1)

    println(@sprintf("Total steps: %d. Spin-up steps: %d, Output every %d steps. Total: %d output frames. Diagnostics every %d steps, max writes per file: %d", 
            nsteps, spinup_step, output_freq, (nsteps - spinup_step) / output_freq, diags_freq, Parameters.max_writes))
    
    println(@sprintf("Total time: %f, Time step: %f, Estimated CFL: %0.3f", Parameters.T, dt, Parameters.cfltune))

    println(@sprintf("Packets: %d. Output every %d steps. Total %d packet frames", Parameters.Npackets, packet_output_freq, (nsteps - spinup_step) / packet_output_freq))
    
    prob = SWQG.Problem(dev; Lx, nx, dt, Parameters.f, Cg=Parameters.Cg, T=Float32, nν, ν, aliased_fraction=1/3, use_filter=false)
    
    set_shafer_initial_condition_QG!(prob, Parameters.Kg, Parameters.ag)

    return prob, nsteps, spinup_step, output_per_packet_freq, packet_output_freq, diags_freq
end

function savepackets!(out, clock, pos, wavenumber, velocity)
    out["p/t/$(clock.step)"] = clock.t
    out["p/x/$(clock.step)"] = Array(pos);
    out["p/k/$(clock.step)"] = Array(wavenumber);
	out["p/u/$(clock.step)"] = Array(velocity);
    return nothing;
end

function savediagnostics(diagnostics, diagnostic_names)
    for (i, diag) in enumerate(diagnostics)
        savediagnostic(diag, diagnostic_names[i], "diagnostics.jld2")
    end
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

function start!()
    # Set up single layer QG problem
    prob, nsteps, spinup_step, output_per_packet_freq, packet_output_freq, diags_freq = initialize_problem()
    SWQG.enforce_reality_condition!(prob)
    
    grid, clock, vars, params = prob.grid, prob.clock, prob.vars, prob.params
    device = grid.device

    # Create initial packets
    Npackets = Parameters.Npackets
    k0 = sqrt(Parameters.ω0^2 - 1) * Parameters.f / Parameters.Cg
    packets = generate_initial_wavepackets(device, Parameters.L, k0, Npackets, Parameters.sqrtNpackets)
    packet_params = (f = Parameters.f, Cg = Parameters.Cg, dt = clock.dt, Npackets = Npackets, k0=k0);

    # Create Output objjects
    filename_func(idx) = @sprintf("%s.%06d.jld2", Parameters.base_filename, idx)
    packet_filename_func(idx) = @sprintf("%s.%06d.jld2", Parameters.packet_base_filename, idx)

    get_sol(prob) = Array(prob.sol)
    output = SequencedOutput(prob, filename_func, (:sol, get_sol), Parameters.max_writes)
    packet_output = SequencedOutput(packet_filename_func, Parameters.packet_max_writes)

    # Configure diagnostics
    KEdiag  = Diagnostic(kinetic_energy, prob; nsteps, freq=diags_freq)
    PEdiag  = Diagnostic(potential_energy, prob; nsteps, freq=diags_freq)
    Ensdiag = Diagnostic(enstrophy, prob; nsteps, freq=diags_freq)
    diags = [KEdiag, PEdiag, Ensdiag]
    diag_names = ["kinetic_energy", "potential_energy", "enstrophy"]

    # Set up memory constructs
    Dev = typeof(grid.device)

    T = typeof(grid.Lx)
    @devzeros Dev T (grid.nx, grid.ny) u1_background v1_background ux1_background uy1_background vx1_background vy1_background
    @devzeros Dev T (grid.nx, grid.ny) u2_background v2_background ux2_background uy2_background vx2_background vy2_background
    
    old_velocity = Velocity(u1_background, v1_background)
    old_grad_v = VelocityGradient(ux1_background, uy1_background, vx1_background, vy1_background)
    
    new_velocity = Velocity(u2_background, v2_background)
    new_grad_v = VelocityGradient(ux2_background, uy2_background, vx2_background, vy2_background)
	
	@devzeros Dev Complex{T} (grid.nkr, grid.nl) temp_device_in_field
    @devzeros Dev T (grid.nx, grid.ny) temp_device_out_field

    @devzeros Dev T (Parameters.Npackets, 2) packet_U
    
    output_u = @views packet_U[:, 1]
    output_v = @views packet_U[:, 2]
    
    packet_pos = @views packets[:, 1:2]

    packet_K = @views packets[:, 3:4]
    packet_k = @views packets[:, 3]
    
    packet_l = @views packets[:, 4]
    
	get_streamfunction(prob) = prob.vars.ψh
    
    get_velocity_info(vars.ψh, grid, packet_params, old_velocity, old_grad_v, temp_device_in_field, temp_device_out_field);
    interpolate_velocity!(old_velocity, packet_pos, grid, output_u, output_v) 

    SequencedOutputs.saveproblem(output)
    SequencedOutputs.saveoutput(output)
    savepackets!(packet_output, clock, packet_pos, packet_K, packet_U); # Save with latest velocity information
    
    startwalltime = time()
    frames = 0:round(Int, nsteps / packet_output_freq)
	packet_frames = 1:round(Int, packet_output_freq)
    ode_template = create_template_ode(packets)

    for step=frames
        if (step % (800 / packet_output_freq) == 0)
            max_udx = max(maximum(abs.(vars.u)) / grid.dx, maximum(abs.(vars.v)) / grid.dy)
            cfl = clock.dt * max_udx
            println(@sprintf("step: %04d, t: %.2f, cfl: %.2e, time: %.2f mins", clock.step, clock.t, cfl, (time() - startwalltime) / 60))
            flush(stdout)
        end        

		old_t = clock.t
        if (clock.step < spinup_step)
            SWQG.stepforward(prob, diags, packet_output_freq * output_per_packet_freq)
            SWQG.updatevars!(prob)
            get_velocity_info(get_streamfunction(prob), grid, packet_params, old_velocity, old_grad_v, temp_device_in_field, temp_device_out_field);
        else 
    		for packet_step=packet_frames
                # Steady background flow
    	        SWQG.stepforward!(prob, diags, 1);
                
                SWQG.updatevars!(prob);
    
                get_velocity_info(get_streamfunction(prob), grid, packet_params, new_velocity, new_grad_v, temp_device_in_field, temp_device_out_field);
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
            
            savepackets!(packet_output, clock, packet_pos, packet_K, packet_U); # Save with latest velocity information
            if (step % output_per_packet_freq == 0)
                SequencedOutputs.saveoutput(output)
            end
        end
    end    
    SequencedOutputs.close(output)
    SequencedOutputs.close(packet_output)
    savediagnostics(diags, diag_names)
end
