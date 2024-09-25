module SWQG
export
  Problem,
  set_solution!,
  enforce_reality_condition!,
  updatevars!,
  energy

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
    ψh  :: Atrans
    qh  :: Atrans
    ζh  :: Atrans
end

function Vars(grid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) ψ q ζ
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) ψh qh ζh

  return Vars(ψ, q, ζ, ψh, qh, ζh)
end

function Problem(dev::Device = CPU();
    nx = 128,
    ny = nx,
    Lx = 2π,
    Ly = Lx,
    ν  = 1.0e-16,
    nν = 4,
    f = 1.0,
    Cg = 1.0,
    stepper = "ETDRK3",
    dt = 5e-2,
    aliased_fraction = 1/3,
    T = Float64)
   
    grid = TwoDGrid(dev; nx, Lx, ny, Ly, aliased_fraction, T)
    params = Params{T}(ν, nν, f, Cg^2/f^2)
    vars = Vars(grid)
    equation = Equation(params, grid)
    return FourierFlows.Problem(equation, stepper, dt, grid, vars, params)
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

    ldiv!(vars.q, grid.rfftplan, deepcopy(vars.qh)) # use deepcopy() because irfft destroys its input
    ldiv!(vars.ψ, grid.rfftplan, deepcopy(vars.ψh)) # use deepcopy() because irfft destroys its input
    ldiv!(vars.ζ, grid.rfftplan, deepcopy(vars.ζh))
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
    
    qhN = N
    @. vars.qh = sol
    streamfunctionfrompv!(vars.ψh, vars.qh, grid, params)

    # Use ζ and ζh as scratch variables
    # Calculate advective terms
    # Use the fact that J(f, g) = (fg_y)_x - (fg_x)_y
    ψ = ζ
    ψh = ζh

    @. ψh = vars.ψh
    ldiv!(vars.ψ, grid.rfftplan, ψh)
    
    ψqyh = ζh
    ψqy = ζ
    @. ψqyh = 1im * grid.l * vars.qh
    ldiv!(ψqy, grid.rfftplan, ψqyh)
    @. ψqy *= vars.ψ
    mul!(ψqyh, grid.rfftplan, ψqy)
    @. qhN += -1im * grid.kr * ψqyh

    ψqxh = ζh
    ψqx = ζ
    @. ψqxh = 1im * grid.kr * vars.qh
    ldiv!(ψqx, grid.rfftplan, ψqxh)
    @. ψqx *= vars.ψ
    mul!(ψqxh, grid.rfftplan, ψqx)
    @. qhN += 1im * grid.l * ψqyh
    return nothing
end

function Equation(params, grid)
    T = eltype(grid)
    dev = grid.device

    # For the standard diagonal problem
    D = @. - params.ν * grid.Krsq^(params.nν)
    L = zeros(dev, T, (grid.nkr, grid.nl))
    L .= D # for u equation
    
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
end