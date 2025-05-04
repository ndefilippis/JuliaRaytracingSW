using Printf;
using FourierFlows
using Random: seed!;
using JLD2;
using LinearAlgebra: ldiv!

import .Parameters;
using .GPURaytracing;
using .TwoLayerQG
using .SequencedOutputs
using .GPURaytracing;

function set_seed_initial_condition!(prob, grid, dev)
    q0 = 1e-2 * device_array(dev)(randn((grid.nx, grid.ny, 2)))
    q0h = rfft(q0, (1, 2))
    
    TwoLayerQG.set_solution!(prob, q0h)
end

function generate_initial_wavepackets(dev, L, k0, Npackets, sqrtNpackets)
    @devzeros typeof(dev) Float64 (Npackets, 4) wavepackets_array
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
    @devzeros typeof(dev) Float64 Npackets frequency_sign
    frequency_sign .= 1
    @views frequency_sign[1:2:end] *= -1 # Make half of the rays negative frequencies
    return wavepackets_array, frequency_sign;
end


function compute_parameters(deformation_radius, intervortex_radius, avg_eddy_velocity, H, f0)
    c₁ = 3.2
    c₂ = 0.36
    l_star = intervortex_radius/deformation_radius

    kappa_star = c₂/log(l_star/c₁)
    U = avg_eddy_velocity / l_star
    μ = 2*U*kappa_star/deformation_radius; # bottom drag
    δb = 4 * f0^2 * deformation_radius^2/H
    return μ, δb, U
end

function initialize_problem()
    dev = GPU()
    Lx=Parameters.L
    nx=Parameters.nx
    dx=Lx/nx
    kmax = nx/2 - 1
    nν=Parameters.nν
    νtune = Parameters.νtune
    H = 1.0
    
    μ, δb, U = compute_parameters(Parameters.deformation_radius, Parameters.intervortex_radius, Parameters.ug, H, Parameters.f)
    δρρ0 = δb / (Parameters.background_Cg/H)
    
    if(μ < 0)
        println("Exiting: μ < 0: ", μ)
        return nothing
    end
    
    cfltune = Parameters.cfltune
    umax = Parameters.ug
    
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

    prob = TwoLayerQG.Problem(dev; stepper="IFMAB3", Lx, nx, dt, f0=Parameters.f, Cg=Parameters.background_Cg, U, δρρ0, T=Float32, nν, ν, μ, aliased_fraction=1/3, use_filter=false)
    
    set_seed_initial_condition!(prob, prob.grid, dev)

    return prob, nsteps, spinup_step, output_per_packet_freq, packet_output_freq, diags_freq
end

function savepacketproblem!(out, params)
    out["params/f0"] = params.f
    out["params/Cg"] = params.Cg
    out["params/dt"] = params.dt
    out["params/N"] = params.Npackets
    out["params/k0"] = params.k0
    out["params/ωsign"] = Array(params.frequency_sign)
end

function write_packets!(out, clock, pos, wavenumber, velocity)
    out["p/t/$(clock.step)"] = clock.t
    out["p/x/$(clock.step)"] = Array(pos);
    out["p/k/$(clock.step)"] = Array(wavenumber);
	out["p/u/$(clock.step)"] = Array(velocity);
    return nothing;
end

function write_packets!(out, clock, pos, wavenumber, velocity, gradient)
    write_packets!(out, clock, pos, wavenumber, velocity)
    out["p/g/$(clock.step)"] = Array(gradient);
    return nothing;
end

function savepacketdata!(out, clock, 
    velocity, gradient, grid, 
    output_u, output_v, 
    output_ux, output_uy, output_vx, output_vy, 
    packet_pos, packet_K, packet_U, packet_grad_U, write_gradients)
    
    interpolate_velocity!(velocity, packet_pos, grid, output_u, output_v)

    if(write_gradients)
        interpolate_gradients!(gradient, packet_pos, grid, output_ux, output_uy, output_vx, output_vy)
        write_packets!(out, clock, packet_pos, packet_K, packet_U, packet_grad_U);
    else
        write_packets!(out, clock, packet_pos, packet_K, packet_U);
    end
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
    seed!(1234)

    prob, nsteps, spinup_step, output_per_packet_freq, packet_output_freq, diags_freq = initialize_problem()
    
    TwoLayerQG.enforce_reality_condition!(prob)
    
    grid, clock, vars, params = prob.grid, prob.clock, prob.vars, prob.params
    device = grid.device

    # Create initial packets
    Npackets = Parameters.Npackets
    k0 = sqrt(Parameters.ω0^2 - Parameters.f^2) / Parameters.background_Cg
    packets, freq_sign = generate_initial_wavepackets(device, Parameters.L, k0, Npackets, Parameters.sqrtNpackets)
    packet_params = (f = Parameters.f, Cg = Parameters.packet_Cg, dt = clock.dt, Npackets = Npackets, k0=k0, frequency_sign = freq_sign);

    # Create Output objects
    filename_func(idx) = @sprintf("%s.%06d.jld2", Parameters.base_filename, idx)
    packet_filename_func(idx) = @sprintf("%s.%06d.jld2", Parameters.packet_base_filename, idx)

    get_sol(prob) = Array(prob.sol)
    output = SequencedOutput(prob, filename_func, (:sol, get_sol), Parameters.max_writes)
    packet_output = SequencedOutput(packet_filename_func, Parameters.packet_max_writes)

    # Configure diagnostics
    KEdiag  = Diagnostic(kinetic_energy, prob; nsteps, freq=diags_freq)
    PEdiag  = Diagnostic(potential_energy, prob; nsteps, freq=diags_freq)
    diags = [KEdiag, PEdiag]
    diag_names = ["kinetic_energy", "potential_energy"]

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
    @devzeros Dev T (Parameters.Npackets, 4) packet_grad_U
    
    output_u = @views packet_U[:, 1]
    output_v = @views packet_U[:, 2]

    output_ux = @views packet_grad_U[:, 1]
    output_uy = @views packet_grad_U[:, 2]
    output_vx = @views packet_grad_U[:, 3]
    output_vy = @views packet_grad_U[:, 4]
    
    packet_pos = @views packets[:, 1:2]

    packet_K = @views packets[:, 3:4]
    packet_k = @views packets[:, 3]
    
    packet_l = @views packets[:, 4]
    
	get_streamfunction(prob) = dropdims(sum(prob.vars.ψh, dims=3), dims=3) # Get barotropic streamfunction
    
    get_velocity_info(get_streamfunction(prob), grid, packet_params, old_velocity, old_grad_v, temp_device_in_field, temp_device_out_field);

    # Initial state
    SequencedOutputs.saveproblem(output)
    SequencedOutputs.saveoutput(output)
    savepacketproblem!(packet_output, packet_params);

    savepacketdata!(packet_output, clock,
        old_velocity, old_grad_v, grid,
        output_u, output_v, output_ux, output_uy, output_vx, output_vy,
        packet_pos, packet_K, packet_U, packet_grad_U, Parameters.write_gradients)
    
    startwalltime = time()
    frames = 0:round(Int, nsteps / packet_output_freq)
	packet_frames = 1:round(Int, packet_output_freq)
    ode_template = create_template_ode(packets) # ODE solver acceleration structure

    println(prob.params)
    local old_t
    local new_t
    old_t = 0.0
    new_t = 0.0
    for step=frames
        if (step % 1 == 0)
            max_udx = max(maximum(abs.(vars.u)) / grid.dx, maximum(abs.(vars.v)) / grid.dy)
            cfl = clock.dt * max_udx
            println(@sprintf("step: %04d, t: %.2f, cfl: %.2e, time: %.2f mins", clock.step, clock.t, cfl, (time() - startwalltime) / 60))
            flush(stdout)
        end        

		old_t = clock.t
        if (clock.step < spinup_step)
            TwoLayerQG.stepforward!(prob, diags, packet_output_freq * output_per_packet_freq)
            TwoLayerQG.updatevars!(prob)
            get_velocity_info(get_streamfunction(prob), grid, packet_params, old_velocity, old_grad_v, temp_device_in_field, temp_device_out_field);
        else 
    		for packet_step=packet_frames
                if(Parameters.use_stationary_background_flow)
                    clock.step += 1
                    clock.t += clock.dt
                else
    	            TwoLayerQG.stepforward!(prob, diags, 1);
                    TwoLayerQG.updatevars!(prob);
                end
                get_velocity_info(get_streamfunction(prob), grid, 
                                      packet_params, new_velocity, new_grad_v, 
                                      temp_device_in_field, temp_device_out_field);
                new_t = clock.t;
                raytrace!(ode_template, old_velocity, new_velocity, old_grad_v, new_grad_v, grid, packets, packet_params.dt, (old_t, new_t), packet_params);
                old_velocity = new_velocity;
                old_grad_v = new_grad_v;
                old_t = new_t;
            end                    

            savepacketdata!(packet_output, clock,
                new_velocity, new_grad_v, grid,
                output_u, output_v, output_ux, output_uy, output_vx, output_vy,
                packet_pos, packet_K, packet_U, packet_grad_U, Parameters.write_gradients)
        end
        if (step % output_per_packet_freq == 0)
            SequencedOutputs.saveoutput(output)
        end
        if(any(isnan.(prob.vars.qh)))
            println("q is nan. Exiting...")
            SequencedOutputs.close(output)
            SequencedOutputs.close(packet_output)
            return
        end
    end    
    SequencedOutputs.close(output)
    SequencedOutputs.close(packet_output)
    savediagnostics(diags, diag_names)
end
