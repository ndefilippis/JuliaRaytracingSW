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