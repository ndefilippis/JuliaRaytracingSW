using GeophysicalFlows, Printf;
using Random: seed!;
using JLD2;

import .Parameters;
import .Raytracing;

function set_initial_condition!(dev, grid, prob, amplitude, nlayers)
   seed!(1234)
   q0  = amplitude * device_array(dev)(randn((grid.nx, grid.ny, nlayers)))
   q0h = prob.timestepper.filter .* rfft(q0, (1, 2))
   q0  = irfft(q0h, grid.nx, (1, 2))
   MultiLayerQG.set_q!(prob, q0) 
end

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

function savepackets!(out, packets::AbstractVector{Raytracing.Wavepacket})
   groupname = "packets"
    jldopen(out.path, "a+") do path
        path["$groupname/t/$(out.prob.clock.step)"] = out.prob.clock.t
        for i=1:size(packets, 1)
            path["$groupname/x/$i/$(out.prob.clock.step)"] = packets[i].x;
            path["$groupname/k/$i/$(out.prob.clock.step)"] = packets[i].k;
        end
    end
    
    return nothing;
end

function get_velocity_info(prob, grid)
    ψh = prob.vars.ψh;
    k = grid.kr;
    l = grid.l;
    uh  = -1im*l.*ψh;
    vh  =  1im*k.*ψh;

    uxh =  1im*k.*uh;
    uyh =  1im*l.*uh;
    vxh =  1im*k.*vh;
    vyh =  1im*l.*vh;
    u = irfft(uh, grid.nx, (1, 2));
    v = irfft(vh, grid.nx, (1, 2));
    ux = irfft(uxh, grid.nx, (1, 2));
    uy = irfft(uyh, grid.nx, (1, 2));
    vx = irfft(vxh, grid.nx, (1, 2));
    vy = irfft(vyh, grid.nx, (1, 2));
    velocity = Raytracing.Velocity(u[:,:,1], v[:,:,1]);
    velocity_gradient = Raytracing.VelocityGradient(ux[:,:,1], uy[:,:,1], vx[:,:,1], vy[:,:,1]);
    return (velocity, velocity_gradient);
end

function simulate!(nsteps, nsubs, npacketsubs, grid, prob, packets, out, diags, packetSpinUpDelay, packet_params)
    saveproblem(out)
    sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

    startwalltime = time()
    frames = 0:round(Int, nsteps / nsubs)

    for j=frames
        if j % (1000 / nsubs) == 0
            cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

            log = @sprintf("step: %04d, t: %.1f, cfl: %.2f, walltime: %.2f min",
                       clock.step, clock.t, cfl, (time()-startwalltime)/60)

            println(log)
        end
        if clock.step >= packetSpinUpDelay
            packet_steps = nsubs / npacketsubs
            for _=1:(nsubs / npacketsubs)
                for _=1:npacketsubs
                    old_v_info = get_velocity_info(prob, grid);
                    old_t = clock.t;

                    stepforward!(prob, diags, 1);
                    MultiLayerQG.updatevars!(prob);

                    new_v_info = get_velocity_info(prob, grid);
                    new_t = clock.t;

                    stepraysforward!(grid, packets, old_v_info, new_v_info, (old_t, new_t), packet_params);
                end
                savepackets!(out, packets);
            end
        else
            stepforward!(prob, diags, nsubs);
            MultiLayerQG.updatevars!(prob);
        end
        saveoutput(out);
    end 
end

function stepraysforward!(grid, packets, v_info_1, v_info_2, tspan, params)
    velocity1, dvelocity1 = v_info_1;
    velocity2, dvelocity2 = v_info_2;
    Raytracing.solve!(velocity1, velocity2, dvelocity1, dvelocity2, grid.x, grid.y, params.Npackets, packets, params.dt, tspan, params);
end

get_sol(prob) = prob.sol # extracts the Fourier-transformed solution

function start!()
    nx, Lx, dt, stepper = Parameters.nx, Parameters.L, Parameters.dt, Parameters.stepper;
    f₀, g, H, ρ, U, μ, β, ν, nν = Parameters.f, Parameters.g, Parameters.H, Parameters.rho, Parameters.U, Parameters.r, Parameters.beta, Parameters.v, Parameters.nv;
    nlayers, Npackets = Parameters.nlayers, Parameters.Npackets
    nsteps, nsubs, npacketsubs, packetSpinUpDelay = Parameters.nsteps, Parameters.nsubs, Parameters.npacketsubs, Parameters.packetSpinUpDelay
    
    dev = CPU();

    prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx, f₀, g, H, ρ, U, μ, β, ν, nν,
                                dt, stepper, aliased_fraction=1/3)

    sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
    x, y = grid.x, grid.y

    E = Diagnostic(MultiLayerQG.energies, prob; nsteps)
    diags = [E]

    filename = joinpath(Parameters.filepath, Parameters.filename)
    if isfile(filename); rm(filename); end
    out = Output(prob, filename, (:sol, get_sol), (:E, MultiLayerQG.energies))

    set_initial_condition!(dev, grid, prob, Parameters.q0_amplitude, nlayers);

    packets = generate_initial_wavepackets(Lx, Parameters.k0Amplitude, Npackets, Parameters.sqrtNpackets);
    packet_params = (f = f₀, Cg = Parameters.Cg, dt = Parameters.packet_dt, Npackets = Npackets);
    simulate!(nsteps, nsubs, npacketsubs, grid, prob, packets, out, diags, Parameters.packetSpinUpDelay, packet_params);
end