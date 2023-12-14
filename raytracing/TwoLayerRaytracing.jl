using GeophysicalFlows, Printf;
using Random: seed!;
using JLD2;
using UnPack;

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

function savepackets!(out, packets::AbstractVector{Raytracing.Wavepacket}, velocity_info)
   groupname = "packets"
    jldopen(out.path, "a+") do path
        path["$groupname/t/$(out.prob.clock.step)"] = out.prob.clock.t
        for i=1:size(packets, 1)
            path["$groupname/x/$i/$(out.prob.clock.step)"] = packets[i].x;
            path["$groupname/k/$i/$(out.prob.clock.step)"] = packets[i].k;
        end
        path["$groupname/rms_U/$(out.prob.clock.step)"] = get_rms_U(velocity_info)
    end
    
    return nothing;
end

function get_velocity_info(prob, grid, params)
    ψh = params.packetVelocityScale * prob.vars.ψh;
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

function get_rms_U(velocity_info::Raytracing.Velocity)
    nx, ny = size(velocity_info.u)
    return sqrt(sum(velocity_info.u.^2 + velocity_info.v.^2)/nx/ny);
end

function simulate!(nsteps, nsubs, npacketsubs, grid, prob, packets, out, packetSpinUpDelay, packet_params)
    saveproblem(out)
    velocity_info, grad_v_info = get_velocity_info(prob, grid, packet_params)
	savepackets!(out, packets, velocity_info);
    sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid

    startwalltime = time()
    frames = 0:round(Int, nsteps / nsubs)

    for j=frames
        if j % (100 / nsubs) == 0
            cfl = clock.dt * maximum([maximum(vars.u) / grid.dx, maximum(vars.v) / grid.dy])

            log = @sprintf("step: %04d, t: %.1f, cfl: %.2f, walltime: %.2f min",
                       clock.step, clock.t, cfl, (time()-startwalltime)/60)

            println(log)
            flush(stdout)
        end
        if clock.step >= packetSpinUpDelay
            packet_steps = nsubs / npacketsubs
            for _=1:(nsubs / npacketsubs)
                old_v_info = get_velocity_info(prob, grid, packet_params);
                old_t = clock.t
                for _=1:npacketsubs
                    stepforward!(prob, [], 1);
                    MultiLayerQG.updatevars!(prob);

                    new_v_info = get_velocity_info(prob, grid, packet_params);
                    new_t = clock.t;

                    @time stepraysforward!(grid, packets, old_v_info, new_v_info, (old_t, new_t), packet_params);

                    old_v_info = new_v_info;
                    old_t = new_t;
                end
				println("Saving packets")
				flush(stdout)
                @time savepackets!(out, packets, old_v_info[1]); # Save with latest velocity information
            end
        else
            stepforward!(prob, [], nsubs);
            MultiLayerQG.updatevars!(prob);
        end
        #saveoutput(out);
    end 
end

function stepraysforward!(grid, packets, v_info_1, v_info_2, tspan, params)
    velocity1, dvelocity1 = v_info_1;
    velocity2, dvelocity2 = v_info_2;
    Raytracing.solve!(velocity1, velocity2, dvelocity1, dvelocity2, grid.x, grid.y, params.Npackets, packets, params.dt, tspan, params);
end

function set_up_problem(filename, stepper)
    L = 2π
    jldopen(filename) do ic_file
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
        pvfromstreamfunction!(prob.sol, ψh, prob.params, prob.grid)
        MultiLayerQG.updatevars!(prob)
        return nx, dt, prob
    end
end

get_streamfunc(prob) = prob.vars.ψh
function modal_energy(prob)
    Eh = prob.grid.Krsq.*abs2.(prob.vars.ψh[:,:,1])
    kr, Ehr = FourierFlows.radialspectrum(Eh, prob.grid)
    return Ehr
end

function start!()
    Lx, stepper = Parameters.L, Parameters.stepper;
    nx, dt, prob = set_up_problem(Parameters.initial_condition_file, stepper);
    
    nsteps, nsubs, npacketsubs, packetSpinUpDelay = Parameters.nsteps, Parameters.nsubs, Parameters.npacketsubs, Parameters.packetSpinUpDelay
    
    sol, clock, params, vars, grid = prob.sol, prob.clock, prob.params, prob.vars, prob.grid
    f, g, H = params.f₀, params.g, params.H

    # E = Diagnostic(MultiLayerQG.energies, prob; nsteps)
    # radialE = Diagnostic(modal_energy, prob; nsteps)
    # diags = [E, radialE]

    filename = joinpath(Parameters.filepath, Parameters.filename)
    if !isdir(Parameters.filepath); mkdir(Parameters.filepath); end
    if isfile(filename); rm(filename); end
    
    out = Output(prob, filename, (:ψh, get_streamfunc))

    # set_initial_condition!(dev, grid, prob, Parameters.q0_amplitude, nlayers);
    Npackets = Parameters.Npackets
    Cg = g*H[1]
    
    packets = generate_initial_wavepackets(Lx, sqrt(Parameters.corFactor^2 - 1)*f/Cg, Npackets, Parameters.sqrtNpackets);
    rms_U = sqrt(sum(vars.u[:,:,1].^2 + vars.v[:,:,1].^2)/nx^2)
    packetVelocityScale = Parameters.initialFroudeNumber * Cg / rms_U
    packet_params = (f = f, Cg = Cg, dt = dt / Parameters.packetStepsPerBackgroundStep, Npackets = Npackets, packetVelocityScale = packetVelocityScale);
    simulate!(nsteps, nsubs, npacketsubs, grid, prob, packets, out, Parameters.packetSpinUpDelay, packet_params);
end
