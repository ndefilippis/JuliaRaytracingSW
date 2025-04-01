module TwoLayerQG

using FourierFlows
using FourierFlows: parsevalsum2
using CUDA

using LinearAlgebra: mul!, ldiv!
using ..IFMAB3

# Only implemented for equal layer heights

struct Params{T} <: AbstractParams
    U :: T         # Background flow velocity
    μ :: T         # Linear bottom drag coefficient
    ν :: T         # Hyperviscosity coefficient
    nν :: Int       # Order of the hyperviscous operator
    F :: T         # Function of Rossby deformation wavenumber
end

struct Vars{Aphys, Atrans} <: AbstractVars
    ψ  :: Aphys
    q  :: Aphys
    ζ  :: Aphys
    u  :: Aphys
    v  :: Aphys
    ψh  :: Atrans
    qh  :: Atrans
    ζh  :: Atrans
    uh  :: Atrans
    vh  :: Atrans
end

function Vars(grid)
    Dev = typeof(grid.device)
    T = eltype(grid)
    
    @devzeros Dev T (grid.nx, grid.ny, 2) ψ q ζ u v
    @devzeros Dev Complex{T} (grid.nkr, grid.nl, 2) ψh qh ζh uh vh
    
    return Vars(ψ, q, ζ, u, v, ψh, qh, ζh, uh, vh)
end

function Problem(dev::Device = CPU();
    nx = 128,
    ny = nx,
    Lx = 2π,
    Ly = Lx,
    ν = 1e-6,
    nν = 4,
    Kd2 = 3.0,
    stepper = "IFMAB3",
    dt = 5e-2,
    aliased_fraction = 1/3,
    T = Float64,
    use_filter=false,
    stepper_kwargs...)

    grid = TwoDGrid(dev; nx, Lx, ny, Ly, aliased_fraction, T)
    params = Params{T}(U, μ, ν, nν, Kd2/2)
    vars = Vars(grid)
    equation = Equation(params, grid)
    if stepper == "IFMAB3"
        clock = FourierFlows.Clock{T}(dt, 0, 0)
        timestepper = IFMAB3TimeStepper(equation, dt, dev; diagonal=true, use_filter, stepper_kwargs...)
        sol = zeros(dev, equation.T, equation.dims)
        return FourierFlows.Problem(sol, clock, equation, grid, vars, params, timestepper)
    else
        return FourierFlows.Problem(equation, stepper, dt, grid, vars, params)
    end
end

function pvfromstreamfunction!(qh, ψh, grid, params)
    ψ1h = @views ψh[:,:,1]
    ψ2h = @views ψh[:,:,2]
    q1h = @views qh[:,:,1]
    q2h = @views qh[:,:,2]
    @. q1h = -grid.Krsq * ψh1 + params.F * (ψh2 - ψh1)
    @. q2h = -grid.Krsq * ψh2 + params.F * (ψh1 - ψh2)
end

function streamfunctionfrompv!(ψh, qh, grid, params)
    ψ1h = @views ψh[:,:,1]
    ψ2h = @views ψh[:,:,2]
    q1h = @views qh[:,:,1]
    q2h = @views qh[:,:,2]
    @. ψ1h = -grid.Krsq * qh1 - params.F * (qh2 + qh1)
    @. ψ2h = -grid.Krsq * qh2 - params.F * (qh1 + qh2)
    @. ψh /= grid.Krsq*(grid.Krsq + 2*params.F)
end

function updatevars!(prob)
    vars, grid, sol, params = prob.vars, prob.grid, prob.sol, prob.params

    dealias!(sol, grid)
    @. vars.qh = sol
    streamfunctionfrompv!(vars.ψh, vars.qh, grid, params)
    @. vars.ζh = -grid.Krsq * vars.ψh
    @. vars.uh = -im * grid.l  * vars.ψh
    @. vars.vh =  im * grid.kr * vars.ψh

    ldiv!(vars.q, grid.rfftplan, deepcopy(vars.qh)) # use deepcopy() because irfft destroys its input
    ldiv!(vars.ψ, grid.rfftplan, deepcopy(vars.ψh)) # use deepcopy() because irfft destroys its input
    ldiv!(vars.ζ, grid.rfftplan, deepcopy(vars.ζh))
    ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh))
    ldiv!(vars.v, grid.rfftplan, deepcopy(vars.vh))
    return nothing
end

function enforce_reality_condition!(prob)
    vars, grid, sol = prob.vars, prob.grid, prob.sol

    dealias!(sol, grid)
    @. vars.qh = sol

    updatevars!(prob)
        
    mul!(vars.qh, grid.rfftplan, deepcopy(vars.q))
    mul!(vars.ψh, grid.rfftplan, deepcopy(vars.ψ))
    return nothing
end

function calcN!(N, sol, t, clock, vars, params, grid)
    dealias!(sol, grid)

    @. vars.qh = sol
    streamfunctionfrompv(vars.ψh, vars.qh, grid, params)
    qh = vars.ζh # Use ζh as a temporary variable
    @. qh = vars.qh
    ldiv!(vars.q, grid.rfftplan, qh)
    
    # Use ζ and ζh as scratch variables
    ψxqh = vars.ζh
    ψxq = vars.ζ
    @. ψxqh = 1im * grid.kr * vars.ψh
    ldiv!(ψxq, grid.rfftplan, ψxqh)
    @. ψxq *= vars.q
    mul!(ψxqh, grid.rfftplan, ψxq)
    @. N = -1im * grid.l * ψxqh

    ψyqh = vars.ζh
    ψyq = vars.ζ
    @. ψyqh = 1im * grid.l * vars.ψh
    ldiv!(ψyq, grid.rfftplan, ψyqh)
    @. ψyq *= vars.q
    mul!(ψyqh, grid.rfftplan, ψyq)
    @. N += 1im * grid.kr * ψyqh

    return nothing
end

function Equation(params, grid)
    T = eltype(grid)
    dev = grid.device

    # For the standard diagonal problem
    D = @. - params.ν * grid.Krsq^(params.nν)
    L = zeros(dev, T, (grid.nkr, grid.nl, 2))
    L .= D          # Hyperviscous dissipation

    L1 = @views L[:, :, 1]
    L2 = @views L[:, :, 2]
    L1 .+= 1im * params.U * grid.kr                 # Add mean-flow advection
    L2 .+= -1im * params.U * grid.kr .- params.μ    # Add bottom drag and mean-flow advection
    return FourierFlows.Equation(L, calcN!, grid)
end

function set_solution!(prob, ψ0h)
    vars, grid, sol = prob.vars, prob.grid, prob.sol

    A = typeof(vars.qh) # determine the type of vars.uh
    pvfromstreamfunction!(vars.qh, A(ψ0h), grid, prob.params)

    @. sol = vars.qh
    updatevars!(prob)

    return nothing
end

@inline kinetic_energy(prob) = kinetic_energy(prob.sol, prob.vars, prob.params, prob.grid)

function kinetic_energy(sol, vars, params, grid)
  streamfunctionfrompv!(vars.ψh, sol, grid, params)
  @. vars.uh = sqrt.(grid.Krsq) * vars.ψh      # vars.uh is a dummy variable

  return parsevalsum2(vars.uh , grid) / (2 * grid.Lx * grid.Ly)
end


@inline potential_energy(prob) = potential_energy(prob.sol, prob.vars, prob.params, prob.grid)

function potential_energy(sol, vars, params, grid)
  streamfunctionfrompv!(vars.ψh, sol, grid, params)

  return params.Kd2 * parsevalsum2(vars.ψh, grid) / (2 * grid.Lx * grid.Ly)
end

@inline energy(prob) = energy(prob.sol, prob.vars, prob.params, prob.grid)

@inline energy(sol, vars, params, grid) = kinetic_energy(sol, vars, params, grid) + potential_energy(sol, vars, params, grid)

function enstrophy(sol, vars, params, grid)
  @. vars.qh = sol
  return parsevalsum2(vars.qh, grid) / (2 * grid.Lx * grid.Ly)
end

@inline enstrophy(prob) = enstrophy(prob.sol, prob.vars, prob.params, prob.grid)

@inline energy_dissipation(prob) = energy_dissipation(prob.sol, prob.vars, prob.params, prob.grid)

@inline function energy_dissipation(sol, vars, params, grid)
  energy_dissipationh = vars.uh # use vars.uh as scratch variable

  @. energy_dissipationh = params.ν * grid.Krsq^(params.nν-1) * abs2(sol)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(energy_dissipationh, grid)
end

@inline enstrophy_dissipation(prob) = enstrophy_dissipation(prob.sol, prob.vars, prob.params, prob.grid)

@inline function enstrophy_dissipation(sol, vars, params, grid)
  enstrophy_dissipationh = vars.uh # use vars.uh as scratch variable

  @. enstrophy_dissipationh = params.ν * grid.Krsq^params.nν * abs2(sol)
  return 1 / (grid.Lx * grid.Ly) * parsevalsum(enstrophy_dissipationh, grid)
end
end