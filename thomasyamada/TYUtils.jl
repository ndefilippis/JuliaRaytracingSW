module TYUtils

using FourierFlows: device_array

export
    compute_balanced_basis,
    compute_wave_bases,
    decompose_balanced_wave

function compute_balanced_basis(grid)
    Φ₀ = device_array(grid.device)(Array{Complex{Float64}, 3}(undef, grid.nkr, grid.nl, 3))
    ω = @. sqrt(1 + grid.kr^2 + grid.l^2)
    @. Φ₀[:,:,1] =  im * grid.l  / ω
    @. Φ₀[:,:,2] = -im * grid.kr / ω
    @. Φ₀[:,:,3] = -1 / ω
    Φ₀[1,1,:] = device_array(grid.device)([0, 0, 1])
    
    return Φ₀
end

function compute_wave_bases(grid)
    ω = @. sqrt(1 + grid.kr^2 + grid.l^2)
    Φ₊ = device_array(grid.device)(Array{Complex{Float64}, 3}(undef, grid.nkr, grid.nl, 3))
    Φ₋ = device_array(grid.device)(Array{Complex{Float64}, 3}(undef, grid.nkr, grid.nl, 3))
    
    @. Φ₊[:,:,1] = (ω*grid.kr + im * grid.l) * sqrt(grid.invKrsq/2)/ω
    @. Φ₊[:,:,2] = (ω*grid.l - im * grid.kr) * sqrt(grid.invKrsq/2)/ω
    @. Φ₊[:,:,3] = (ω^2 - 1) * sqrt(grid.invKrsq/2)/ω
    Φ₊[1,1,:] = device_array(grid.device)([im, 1, 0]/sqrt(2))
    
    @. Φ₋[:,:,1] = (-ω*grid.kr + im * grid.l) * sqrt(grid.invKrsq/2)/ω
    @. Φ₋[:,:,2] = (-ω*grid.l - im * grid.kr) * sqrt(grid.invKrsq/2)/ω
    @. Φ₋[:,:,3] = (ω^2 - 1) * sqrt(grid.invKrsq/2)/ω
    Φ₋[1,1,:] = device_array(grid.device)([im, -1, 0]/sqrt(2))
   
    return (Φ₊, Φ₋)
end

function decompose_balanced_wave2(solution, grid)
    baroclinic_components = solution[:,:,2:4]
    uc = baroclinic_components[:,:,1]
    vc = baroclinic_components[:,:,2] 
end

function decompose_balanced_wave(solution, grid)
    Φ₀ = compute_balanced_basis(grid)
    Φ₊, Φ₋ = compute_wave_bases(grid)
    baroclinic_components = solution[:,:,2:4]
    Gh = sum(baroclinic_components .* conj(Φ₀), dims=3) .* Φ₀
    Wh = sum(baroclinic_components .* conj(Φ₊), dims=3) .* Φ₊ + sum(baroclinic_components .* conj(Φ₋), dims=3) .* Φ₋
    return (Gh, Wh)
end
end
