module SWQG
export
    Problem,
    set_solution!,
    enforce_reality_condition!,
    updatevars!,
    energy,
    kinetic_energy,
    potential_energy,
    enstrophy

using FourierFlows
using FourierFlows: parsevalsum2
using CUDA

using LinearAlgebra: mul!, ldiv!
using ..IFMAB3

struct Params{T} <: AbstractParams
    ν :: T         # Hyperviscosity coefficient
    nν :: Int       # Order of the hyperviscous operator
    Kd2 :: T         # Deformation wavenumber squared
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

struct StochasticVars{Aphys, Atrans} <: AbstractVars
    ψ  :: Aphys
    q  :: Aphys
    ζ  :: Aphys
    ψh  :: Atrans
    qh  :: Atrans
    ζh  :: Atrans
    Fh  :: Atrans
end

function StochasticVars(grid)
    Dev = typeof(grid.device)
    T = eltype(grid)
    
    @devzeros Dev T (grid.nx, grid.ny) ψ q ζ
    @devzeros Dev Complex{T} (grid.nkr, grid.nl) ψh qh ζh Fh
    
    return Vars(ψ, q, ζ, ψh, qh, ζh, Fh)
end

function Vars(grid)
    Dev = typeof(grid.device)
    T = eltype(grid)
    
    @devzeros Dev T (grid.nx, grid.ny) ψ q ζ u v
    @devzeros Dev Complex{T} (grid.nkr, grid.nl) ψh qh ζh uh vh
    
    return Vars(ψ, q, ζ, u, v, ψh, qh, ζh, uh, vh)
end

nothingfunction(args...) = nothing

function Problem(dev::Device = CPU();
    nx = 128,
    ny = nx,
    Lx = 2π,
    Ly = Lx,
    ν  = 1.0e-16,
    nν = 4,
    f = 1.0,
    Cg = 1.0,
    stepper = "IFMAB3",
    dt = 5e-2,
    calcF! = nothingfunction,
    aliased_fraction = 1/3,
    T = Float64,
    use_filter=false,
    stepper_kwargs...)
   
    grid = TwoDGrid(dev; nx, Lx, ny, Ly, aliased_fraction, T)
    params = Params{T}(ν, nν, f^2/Cg^2)
    vars = calcF! == nothingfunction ? Vars(grid) : StochasticVars(grid)
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
    @. qh = -(grid.Krsq + params.Kd2) * ψh
end

function streamfunctionfrompv!(ψh, qh, grid, params)
    @. ψh = -qh / (grid.Krsq + params.Kd2)
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
    streamfunctionfrompv!(vars.ψh, vars.qh, grid, params)
    qh = vars.ζh
    @. qh = vars.qh
    ldiv!(vars.q, grid.rfftplan, qh)

    # Use ζ and ζh as scratch variables
    # Calculate advective terms
    # q_t = -J(ψ, q)
    # Use the fact that J(f, g) = (f_xg)_y - (f_yg)_x
    ψxqh = vars.ζh
    ψxq = vars.ζ
    @. ψxqh = 1im * grid.kr * vars.ψh
    ldiv!(ψxq, grid.rfftplan, ψxqh) # Contains qh_y
    @. ψxq *= vars.q
    mul!(ψxqh, grid.rfftplan, ψxq) # Contains (ψq_y)h_x
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

addforcing!(N, sol, t, clock, vars::Vars, params, grid) = nothing

function addforcing!(N, sol, t, clock, vars::StochasticVars, params, grid)
  params.calcF!(vars.Fh, sol, t, clock, vars, params, grid)
  
  @. N += vars.Fh
  return nothing
end

function Equation(params, grid)
    T = eltype(grid)
    dev = grid.device

    # For the standard diagonal problem
    D = @. - params.ν * grid.Krsq^(params.nν)
    L = zeros(dev, T, (grid.nkr, grid.nl))
    L .= D

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