const u_INDEX = 1
const v_INDEX = 2
const η_INDEX = 3

function wave_balanced_decomposition(prob)
    return wave_balanced_decomposition(prob.vars.uh, prob.vars.vh, prob.vars.ηh, prob.grid, prob.params)
end

function wave_balanced_decomposition(uh, vh, ηh, grid, params)
    Kd2 = params.f^2/params.Cg2
    qh = @. 1im * grid.kr * vh - 1im * grid.l * uh - params.f * ηh
    ψh = @. -qh / (grid.Krsq + Kd2)
    ugh = -1im * grid.l  .* ψh
    vgh =  1im * grid.kr .* ψh
    ηgh = params.f/params.Cg2 * ψh
    uwh = uh - ugh
    vwh = vh - vgh
    ηwh = ηh - ηgh
    return ((ugh, vgh, ηgh), (uwh, vwh, ηwh))
end

function compute_balanced_wave_bases(grid, params)
    Φ₀ = device_array(grid.device)(Array{Complex{Float64}, 3}(undef, grid.nkr, grid.nl, 3))
    Φ₊ = device_array(grid.device)(Array{Complex{Float64}, 3}(undef, grid.nkr, grid.nl, 3))
    Φ₋ = device_array(grid.device)(Array{Complex{Float64}, 3}(undef, grid.nkr, grid.nl, 3))

    Cg = sqrt(params.Cg2)
    ω = @. sqrt(params.f^2 + params.Cg2*grid.Krsq)
    
    @. Φ₀[:,:,u_INDEX] = -1im * grid.l  * Cg / ω
    @. Φ₀[:,:,v_INDEX] =  1im * grid.kr * Cg / ω
    @. Φ₀[:,:,η_INDEX] =  params.f / ω
    Φ₀[1,1,:] = device_array(grid.device)([0, 0, 1])

    @. Φ₊[:,:,u_INDEX] = @. (ω*grid.kr + 1im * params.f*grid.l ) * sqrt(grid.invKrsq/2)/ω
    @. Φ₊[:,:,v_INDEX] = @. (ω*grid.l  - 1im * params.f*grid.kr) * sqrt(grid.invKrsq/2)/ω
    @. Φ₊[:,:,η_INDEX] = @. Cg * grid.Krsq * sqrt(grid.invKrsq/2)/ω
    Φ₊[1,1,:] = device_array(grid.device)([im, 1, 0]/sqrt(2))
    
    @. Φ₋[:,:,u_INDEX] = @. (-ω*grid.kr + 1im * params.f*grid.l ) * sqrt(grid.invKrsq/2)/ω
    @. Φ₋[:,:,v_INDEX] = @. (-ω*grid.l  - 1im * params.f*grid.kr) * sqrt(grid.invKrsq/2)/ω
    @. Φ₋[:,:,η_INDEX] = @. Cg * grid.Krsq * sqrt(grid.invKrsq/2)/ω
    Φ₋[1,1,:] = device_array(grid.device)([-im, 1, 0]/sqrt(2))

    return Φ₀, Φ₊, Φ₋
end

function compute_balanced_wave_weights(uh, vh, ηh, Φ₀, Φ₊, Φ₋)
    c₀ = uh.*conj(Φ₀[:,:,u_INDEX]) + vh.*conj(Φ₀[:,:,v_INDEX]) + ηh.*conj(Φ₀[:,:,η_INDEX])
    c₊ = uh.*conj(Φ₊[:,:,u_INDEX]) + vh.*conj(Φ₊[:,:,v_INDEX]) + ηh.*conj(Φ₊[:,:,η_INDEX])
    c₋ = uh.*conj(Φ₋[:,:,u_INDEX]) + vh.*conj(Φ₋[:,:,v_INDEX]) + ηh.*conj(Φ₋[:,:,η_INDEX])
    return c₀, c₊, c₋
end