module RSWRaytracingDriver

using Printf;
using FourierFlows
using Random: seed!;
using JLD2;
using LinearAlgebra: ldiv!

import ..Parameters;
using ..RotatingShallowWater
using ..GPURaytracing

export create_raytracing_model, set_initial_condition!, get_streamfunction!, estimate_max_U, create_fourier_flows_problem

function set_initial_condition!(prob, grid, dev)
    dev = typeof(grid.device)
    T = typeof(grid.Lx)
    
    Kg = Parameters.Kg
    Kw = Parameters.Kw
    ag = Parameters.ag
    aw = Parameters.aw

    Cg2 = prob.params.Cg2
    f = prob.params.f

    @devzeros dev Complex{T} (grid.nkr, grid.nl) ugh vgh ηgh uwh vwh ηwh
    @devzeros dev T (grid.nx, grid.ny) ug uw
    
    geo_filter  = Kg[1]^2 .<= grid.Krsq .<= Kg[2]^2
    wave_filter = (Kw[1]^2 .<= grid.Krsq .<= Kw[2]^2) .& (grid.Krsq .> 0)
    phase = device_array(grid.device)(2π*rand(grid.nkr, grid.nl))
    sgn =  device_array(grid.device)(sign.(rand(grid.nkr, grid.nl) .- 0.5))
    shift = exp.(1im * phase)
    ηgh[geo_filter] += ( 0.5   * shift)[geo_filter]
    ugh[geo_filter] += (-0.5im * Cg2 / f * grid.l  .* shift)[geo_filter]
    vgh[geo_filter] += ( 0.5im * Cg2 / f * grid.kr .* shift)[geo_filter]

    ldiv!(ug, grid.rfftplan, deepcopy(ugh))
    ηgh *= ag / maximum(abs.(ug))
    ugh *= ag / maximum(abs.(ug))
    vgh *= ag / maximum(abs.(ug))

    ωK =  sgn .* sqrt.(f^2 .+ Cg2 * grid.Krsq)
    ηwh[wave_filter] += (0.5 * shift)[wave_filter]
    uwh[wave_filter] += (grid.invKrsq.*(0.5 * grid.kr .* ωK .* shift + 0.5im * f * grid.l  .* shift))[wave_filter]
    vwh[wave_filter] += (grid.invKrsq.*(0.5 * grid.l .*  ωK .* shift - 0.5im * f * grid.kr .* shift))[wave_filter]
    
    ldiv!(uw, grid.rfftplan, deepcopy(uwh))
    ηwh *= aw / maximum(abs.(uw))
    uwh *= aw / maximum(abs.(uw))
    vwh *= aw / maximum(abs.(uw))
    RotatingShallowWater.set_solution!(prob, ugh + uwh, vgh + vwh, ηgh + ηwh)
end

function get_streamfunction!(ψh, prob)
    uh = @views prob.sol[:,:,1]
    vh = @views prob.sol[:,:,2]
    ηh = @views prob.sol[:,:,3]
    return get_streamfunction!(ψh, uh, vh, ηh, prob.params, prob.grid)
end

function get_streamfunction!(ψh, uh, vh, ηh, params, grid)
    Kd2 = params.f^2/params.Cg2
    @. ψh = 1im * grid.kr * vh - 1im * grid.l * uh - params.f * ηh
    @. ψh /= -(grid.Krsq + Kd2)
end

function estimate_max_U()
    return Parameters.ag + Parameters.aw
end

function create_fourier_flows_problem(dev, common_params)
     return RotatingShallowWater.Problem(dev; f=Parameters.f, Cg=Parameters.Cg, common_params...)
end

end
