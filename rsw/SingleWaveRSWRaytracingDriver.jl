using Printf;
using FourierFlows
using Random: seed!;
using JLD2;
using CUDA;
using LinearAlgebra: ldiv!, mul!;

import .Parameters;
using .GPURaytracing;
using .SequencedOutputs;
using .RSWRaytracingDriver;
using .RaytracingDriver;
using .RotatingShallowWater;

export start_raytracing!

function generate_initial_wavepacket_xk(Npackets, grid)
    dev = tyopeof(grid.device)
    T = typeof(grid.Lx)

    sqrtNpackets = floor(Int, sqrt(Npackets))
    
    @devzeros dev Float64 (Npackets, 2) x0 y0 k0 l0
    @devzeros dev Int (Npackets, 2) k0_idx l0_idx
    θ = 2π*(0:(Npackets-1))/Npackets
    K0 = Parameters.K0

    @. k0 = @. round(Int, abs(K0 * cos(θ)))
    @. l0 = @. round(Int, K0 * sin(θ))

    l0_idx = @. mod(l0, grid.nl) + 1
    k0_idx = @. k0 + 1;
    
    I = device_array(dev)(1:sqrtNpackets)
    J = device_array(dev)(1:Npackets)
    diagonal = @. I*L/sqrtNpackets - L/2 - offset
    phase = 2*pi*J/Npackets
    x0 .= repeat(diagonal, outer=sqrtNpackets)
    y0 .= repeat(diagonal, inner=sqrtNpackets)
    @devzeros typeof(dev) Float64 Npackets frequency_sign
    frequency_sign .= 1
    @views frequency_sign[1:2:end] *= -1 # Make half of the rays negative frequencies

    return x0, y0, k0, l0, k0_idx, l0_idx, frequency_sign
end

function envelope(x0, y0, env_size, grid)
    mod_x = @. mod(grid.x - x0 - grid.x[1], grid.Lx) + grid.x[1]
    mod_y = @. mod(grid.y' - y0 - grid.y[1], grid.Ly) + grid.y[1]
    return @. exp(-(mod_x/env_size)^2 - (mod_y/env_size)^2)
end

function create_single_wave_envelope(grid, x0, y0, k0_idx, l0_idx, phase, sign, env_size, aw, params)
    dev = typeof(grid.device)
    T = typeof(grid.Lx)
    
    @devzeros dev Complex{T} (grid.nkr, grid.nl) uwh vwh ηwh
    @devzeros dev T (grid.nx, grid.ny) uw vw ηw

    A = typeof(uwh)

    CUDA.@allowscalar k0 = grid.kr[k0_idx]
    CUDA.@allowscalar l0 = grid.l[l0_idx]
    Ksq = k0^2 + l0^2
    invKsq = 1/Ksq
    ωK = sqrt.(params.f^2 .+ params.Cg2 * Ksq)
    env = envelope(x0, y0, env_size, grid) 
    waveform = @. env * exp(1im * k0 * grid.x + 1im * l0 * grid.y' + 1im*phase)
    waveform = A(waveform)
    
    @. ηw = real(0.5 * waveform)
    @. uw = real(invKsq.*(0.5 * k0 .* ωK .+ 0.5im * params.f * l0) * waveform)
    @. vw = real(invKsq.*(0.5 * l0 .* ωK .- 0.5im * params.f * k0) * waveform)

    max_u = maximum(abs.(uw))
    @. ηw *= aw / max_u
    @. uw *= aw / max_u
    @. vw *= aw / max_u
    
    mul!(uwh, grid.rfftplan, uw)
    mul!(vwh, grid.rfftplan, vw)
    mul!(ηwh, grid.rfftplan, ηw)
    
    return uwh, vwh, ηwh
end

function add_single_wave_to_solution(prob)
    params, grid = prob.params, prob.grid
    uh = prob.vars.uh
    vh = prob.vars.vh
    ηh = prob.vars.ηh
    
    Kd2 = params.f^2/params.Cg2
    qh = @. 1im * grid.kr * vh - 1im * grid.l * uh - params.f * ηh
    ψh = @. -qh / (grid.Krsq + Kd2)
    ugh = -1im * grid.l  .* ψh
    vgh =  1im * grid.kr .* ψh
    ηgh = params.f/params.Cg2 * ψh

    uwh, vwh, ηwh = create_single_wave_envelope(grid, 
        Parameters.x0, Parameters.y0, 
        Parameters.k0_idx, Parameters.l0_idx, 
        Parameters.phase, Parameters.sign, Parameters.env_size, Parameters.aw_2, params)
    RotatingShallowWater.set_solution!(prob, ugh + uwh, vgh + vwh, ηgh + ηwh)
end

function add_wavepackets_to_solution(prob, x0, y0, k0_idx, l0_idx, sign)
    params, grid = prob.params, prob.grid
    uh = prob.vars.uh
    vh = prob.vars.vh
    ηh = prob.vars.ηh

    Kd2 = params.f^2/params.Cg2
    qh = @. 1im * grid.kr * vh - 1im * grid.l * uh - params.f * ηh
    ψh = @. -qh / (grid.Krsq + Kd2)
    ugh = -1im * grid.l  .* ψh
    vgi 1im * grid.kr .* ψh
    ηgh = params.f/params.Cg2 * ψh
    
    dev = typeof(grid.device)
    T = typeof(grid.Lx)
    
    @devzeros dev Complex{T} (grid.nkr, grid.nl) uwh vwh ηwh

    for i=1:Parameters.Npackets
        CUDA.@allowscalar uwh_i, vwh_i, ηwh_i = create_single_wave_envelope(grid, x0[i], y0[i], k0_idx[i], l0_idx[i], 
            0, sign[i], Parameters.env_size, Parameters.aw_2, params)
        uwh += uwh_i
        vwh += vwh_i
        ηwh += ηwh_i
    end
end

function generate_single_wavepacket(dev, x0, k0)
    @devzeros dev Float64 (1, 4) wavepackets_array
    @views wavepackets_array[1,1:2] = x0
    @views wavepackets_array[1,3:4] = k0
    
    @devzeros dev Float64 (1, 1) frequency_sign
    frequency_sign .= 1
    return wavepackets_array, frequency_sign;
end

function generate_wavepackets_from_parameters(dev, Npackets, x0, k0, freq_sign)
    @devzeros dev Float64 (Npackets, 4) wavepackets_array
    @views wavepackets_array[:,1:2] = x0
    @views wavepackets_array[:,3:4] = k0

    A = typeof(wavepackets_array)
    frequency_sign = A(freq_sign)
    return wavepackets_array, frequency_sign
end

function start!()
    seed!(1235)

    prob, nsteps, spinup_step, packet_spinup_step, output_per_packet_freq, packet_output_freq, diags_freq = RaytracingDriver.initialize_problem(1, Parameters.ag)
    RotatingShallowWater.enforce_reality_condition!(prob)
    
    grid, clock, vars, params = prob.grid, prob.clock, prob.vars, prob.params
    device = grid.device
    Dev = typeof(grid.device)
    T = typeof(grid.Lx)

    # Create initial packets
    Npackets = 1
    @devzeros Dev Float64 (1, 2) x0 k0
    @. x0 = 0
    CUDA.@allowscalar k0[1] = grid.kr[Parameters.k0_idx]
    CUDA.@allowscalar k0[2] = grid.l[Parameters.l0_idx]
    packets, freq_sign = generate_single_wavepacket(Dev, x0, k0)
    packet_params = (f = Parameters.f, Cg = Parameters.packet_Cg, dt = clock.dt, Npackets = Npackets, k0=k0, frequency_sign = freq_sign);

    # Create Output objects
    filename_func(idx) = @sprintf("%s.%06d.jld2", Parameters.base_filename, idx)
    packet_filename_func(idx) = @sprintf("%s.%06d.jld2", Parameters.packet_base_filename, idx)

    get_sol(prob) = Array(prob.sol)
    output = SequencedOutputs.SequencedOutput(prob, filename_func, (:sol, get_sol), Parameters.max_writes)
    packet_output = SequencedOutputs.SequencedOutput(packet_filename_func, Parameters.packet_max_writes)

    # Configure diagnostics
    KEdiag  = Diagnostic(RotatingShallowWater.kinetic_energy, prob; nsteps, freq=diags_freq)
    PEdiag  = Diagnostic(RotatingShallowWater.potential_energy, prob; nsteps, freq=diags_freq)
    diags = [KEdiag, PEdiag]
    diag_names = ["kinetic_energy", "potential_energy"]

    # Set up memory constructs
    FourierFlows.@devzeros Dev T (grid.nx, grid.ny) u1_background v1_background ux1_background uy1_background vx1_background vy1_background
    FourierFlows.@devzeros Dev T (grid.nx, grid.ny) u2_background v2_background ux2_background uy2_background vx2_background vy2_background
    
    old_velocity = GPURaytracing.Velocity(u1_background, v1_background)
    old_grad_v = GPURaytracing.VelocityGradient(ux1_background, uy1_background, vx1_background, vy1_background)
    
    new_velocity = GPURaytracing.Velocity(u2_background, v2_background)
    new_grad_v = GPURaytracing.VelocityGradient(ux2_background, uy2_background, vx2_background, vy2_background)
	
	@devzeros Dev Complex{T} (grid.nkr, grid.nl) temp_device_in_field
    @devzeros Dev Complex{T} (grid.nkr, grid.nl) streamfunction
    @devzeros Dev T (grid.nx, grid.ny) temp_device_out_field

    @devzeros Dev T (1, 2) packet_U
    @devzeros Dev T (1, 4) packet_grad_U
    
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
    
    get_streamfunction!(streamfunction, prob)
    get_velocity_info(streamfunction, prob, packet_params, old_velocity, old_grad_v, temp_device_in_field, temp_device_out_field);

    # Initial state
    SequencedOutputs.saveproblem(output)
    SequencedOutputs.saveoutput(output)
    savepacketproblem!(packet_output, packet_params);

    savepacketdata!(packet_output, clock,
        old_velocity, old_grad_v, grid,
        output_u, output_v, output_ux, output_uy, output_vx, output_vy,
        packet_pos, packet_K, packet_U, packet_grad_U, Parameters.write_gradients)
    
    startwalltime = time()
    startup_frames = 0:round(Int, packet_spinup_step / packet_output_freq)
    analysis_frames = 0:round(Int, (nsteps - packet_spinup_step) / packet_output_freq)
	packet_frames = 1:round(Int, packet_output_freq)
    ode_template = GPURaytracing.create_template_ode(packets) # ODE solver acceleration structure

    local old_t
    local new_t
    old_t = 0.0
    new_t = 0.0
    for step=startup_frames
        if (step % 10 == 0)
            max_udx = max(maximum(abs.(vars.u)) / grid.dx, maximum(abs.(vars.v)) / grid.dy)
            cfl = clock.dt * max_udx
            println(@sprintf("step: %04d, t: %.2f, cfl: %.2e, time: %.2f mins", clock.step, clock.t, cfl, (time() - startwalltime) / 60))
            flush(stdout)
        end        

		old_t = clock.t
        stepforward!(prob, diags, packet_output_freq * output_per_packet_freq)
        updatevars!(prob)
        get_streamfunction!(streamfunction, prob)
        get_velocity_info(streamfunction, prob, packet_params, old_velocity, old_grad_v, temp_device_in_field, temp_device_out_field);
    end

    add_single_wave_to_solution(prob)
    savepacketdata!(packet_output, clock,
            new_velocity, new_grad_v, grid,
            output_u, output_v, output_ux, output_uy, output_vx, output_vy,
            packet_pos, packet_K, packet_U, packet_grad_U, Parameters.write_gradients)
    SequencedOutputs.saveoutput(output)
    for step=analysis_frames
        if (step % 10 == 0)
            max_udx = max(maximum(abs.(vars.u)) / grid.dx, maximum(abs.(vars.v)) / grid.dy)
            cfl = clock.dt * max_udx
            println(@sprintf("step: %04d, t: %.2f, cfl: %.2e, time: %.2f mins", clock.step, clock.t, cfl, (time() - startwalltime) / 60))
            flush(stdout)
        end  
        for packet_step=packet_frames
            stepforward!(prob, diags, 1);
            updatevars!(prob);
            get_streamfunction!(streamfunction, prob)
            get_velocity_info(streamfunction, prob, 
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
        SequencedOutputs.saveoutput(output)
        if(any(isnan.(prob.sol)))
            println("Solution is nan. Exiting...")
            SequencedOutputs.close(output)
            SequencedOutputs.close(packet_output)
            return
        end
    end    
    SequencedOutputs.close(output)
    SequencedOutputs.close(packet_output)
    savediagnostics(diags, diag_names)
end
