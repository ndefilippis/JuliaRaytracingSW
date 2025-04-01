module RotatingShallowWater
export
  Problem,
  set_solution!,
  enforce_reality_condition!,
  updatevars!,
  energy,
  wave_balanced_decomposition

using FourierFlows
using FourierFlows: parsevalsum2
using CUDA

using LinearAlgebra: mul!, ldiv!
using ..IFMAB3

struct Params{T} <: AbstractParams
   ν :: T         # Hyperviscosity coefficient
  nν :: Int       # Order of the hyperviscous operator
   f :: T         # Coriolis parameter
 Cg2 :: T         # Group velocity speed squared
end

struct Vars{Aphys, Atrans} <: AbstractVars
    u  :: Aphys
    v  :: Aphys
    η  :: Aphys
    ζ  :: Aphys
    vh  :: Atrans
    uh  :: Atrans
    ηh  :: Atrans
    ζh  :: Atrans
end

struct StochasticVars{Aphys, Atrans} <: AbstractVars
    u  :: Aphys
    v  :: Aphys
    η  :: Aphys
    ζ  :: Aphys
    vh  :: Atrans
    uh  :: Atrans
    ηh  :: Atrans
    ζh  :: Atrans
    Fh  :: Atrans
end

function Vars(grid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) u v η ζ
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) uh vh ηh ζh 

  return Vars(u, v, η, ζ, uh, vh, ηh, ζh)
end

function StochasticVars(grid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) u v η ζ
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) uh vh ηh ζh Fh

  return StochasticVars(u, v, η, ζ, uh, vh, ηh, ζh, Fh)
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
    params = Params{T}(ν, nν, f, Cg^2)
    vars = calcF! == nothingfunction ? Vars(grid) : StochasticVars(grid)
    equation = Equation(params, grid)
    if stepper == "IFMAB3"
        clock = FourierFlows.Clock{T}(dt, 0, 0)
        timestepper = IFMAB3TimeStepper(equation, dt, dev; use_filter, stepper_kwargs...)
        sol = zeros(dev, equation.T, equation.dims)
        return FourierFlows.Problem(sol, clock, equation, grid, vars, params, timestepper)
    else
        return FourierFlows.Problem(equation, stepper, dt, grid, vars, params, stepper_kwargs...)
    end
end

function updatevars!(prob)
    vars, grid, sol = prob.vars, prob.grid, prob.sol

    dealias!(sol, grid)
    @. vars.uh = @view sol[:,:,1]
    @. vars.vh = @view sol[:,:,2]
    @. vars.ηh = @view sol[:,:,3]
    @. vars.ζh = 1im * grid.kr * vars.vh - 1im * grid.l * vars.uh - prob.params.f * vars.ηh

    ldiv!(vars.u, grid.rfftplan, deepcopy(vars.uh)) # use deepcopy() because irfft destroys its input
    ldiv!(vars.v, grid.rfftplan, deepcopy(vars.vh)) # use deepcopy() because irfft destroys its input
    ldiv!(vars.η, grid.rfftplan, deepcopy(vars.ηh)) # use deepcopy() because irfft destroys its input
    ldiv!(vars.ζ, grid.rfftplan, deepcopy(vars.ζh)) # use deepcopy() because irfft destroys its input
    
    return nothing
end

function enforce_reality_condition!(prob)
    vars, grid, sol = prob.vars, prob.grid, prob.sol

    dealias!(sol, grid)
    @. vars.uh = @view sol[:,:,1]
    @. vars.vh = @view sol[:,:,2]
    @. vars.ηh = @view sol[:,:,3]

    updatevars!(prob)
        
    mul!(vars.uh, grid.rfftplan, deepcopy(vars.u))
    mul!(vars.vh, grid.rfftplan, deepcopy(vars.v))
    mul!(vars.ηh, grid.rfftplan, deepcopy(vars.η))
    
    return nothing
end

function NOPcalcN!(N, sol, t, clock, vars, params, grid)
    N .= 0
    return nothing
end

function calcN!(N, sol, t, clock, vars, params, grid)
    dealias!(sol, grid)
    
    uhN = @view N[:,:,1]
    vhN = @view N[:,:,2]
    ηhN = @view N[:,:,3]
    
    @. vars.uh = @view sol[:,:,1]
    @. vars.vh = @view sol[:,:,2]
    @. vars.ηh = @view sol[:,:,3]
    
    # Calculate advective terms
    # Compute real-space u
    # Use ζ and ζh as temp variables
    uh = vars.ζh
    @. uh = vars.uh
    ldiv!(vars.u, grid.rfftplan, uh)

    # and real-space v
    vh = vars.ζh
    @. vh = vars.vh
    ldiv!(vars.v, grid.rfftplan, vh)

    #===
    Compute u * ux and v * vy terms
    ===#

    # u * ux term
    uux = vars.ζ
    uuxh = vars.ζh
    @. uuxh = 1im * grid.kr * vars.uh
    ldiv!(uux, grid.rfftplan, uuxh)
    @. uux *= vars.u
    mul!(uuxh, grid.rfftplan, uux)
    @. uhN = -uuxh

    # v * vy term
    vvy = vars.ζ
    vvyh = vars.ζh
    @. vvyh = 1im * grid.l * vars.vh
    ldiv!(vvy, grid.rfftplan, vvyh)
    @. vvy *= vars.v
    mul!(vvyh, grid.rfftplan, vvy)
    @. vhN = -vvyh

    # ===
    # Compute v * uy and u * vx terms
    # First, need to compute uy and vx, using ζ and ζh as scratch variables
    # ===
    
    # v * uy term
    vuy = vars.ζ
    vuyh  = vars.ζh
    @. vuyh = 1im * grid.l * vars.uh # Store uy
    ldiv!(vuy, grid.rfftplan, vuyh)   # Convert to real space
    @. vuy *= vars.v                 # Multiply by v
    mul!(vuyh, grid.rfftplan, vuy)   # Convert back to spectral space
    @. uhN += -vuyh

    # u * v 
    uvx = vars.ζ
    uvxh  = vars.ζh
    @. uvxh = 1im * grid.kr * vars.vh # Store vx
    ldiv!(uvx, grid.rfftplan, uvxh)
    @. uvx *= vars.u
    mul!(uvxh, grid.rfftplan, uvx)
    @. vhN += -uvxh

    #===
    Compute (ηu)_x and (ηv)_y terms
    ===#

    ηh = vars.ζh
    @. ηh = vars.ηh
    ldiv!(vars.η, grid.rfftplan, ηh)
    
    ηux  = vars.ζ
    ηuxh = vars.ζh
    @. ηux = vars.u .* vars.η
    mul!(ηuxh, grid.rfftplan, ηux)
    @. ηhN = -1im * grid.kr * ηuxh

    ηvy  = vars.ζ
    ηvyh = vars.ζh
    @. ηvy = vars.v .* vars.η
    mul!(ηvyh, grid.rfftplan, ηvy)
    @. ηhN += -1im * grid.l * ηvyh

    addforcing!(N, sol, t, clock, vars, params, grid)
    return nothing
end

addforcing!(N, sol, t, clock, vars::Vars, params, grid) = nothing

function addforcing!(N, sol, t, clock, vars::StochasticVars, params, grid)
  params.calcF!(vars.Fh, sol, t, clock, vars, params, grid)
  
  @. N += vars.Fh
  
  return nothing
end

function Lop_kernel(result, k, l, Nx, Ny, D, f, Cg2)
    i = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    j = blockDim().y * (blockIdx().y - 1) + threadIdx().y
    if i > Nx || j > Ny
        return
    end
    result[i, j, 1, 1] = D[i,j]
    result[i, j, 1, 2] =  f
    result[i, j, 1, 3] = -1im*k[i]*Cg2

    result[i, j, 2, 1] = -f
    result[i, j, 2, 2] = D[i,j]
    result[i, j, 2, 3] = -1im*l[j]*Cg2

    result[i, j, 3, 1] = -1im*k[i]
    result[i, j, 3, 2] = -1im*l[j]
    result[i, j, 3, 3] = D[i,j]
    return
end

function populate_L!(L, grid, params, dev::CPU)
    D = @. - params.ν * grid.Krsq^(params.nν)
    Lop(k, l, i, j) = [D[i,j]          params.f  -1im*k*params.Cg2;
                 -params.f  D[i,j]         -1im*l*params.Cg2;
                 -1im*k     -1im*l     D[i,j]]
    for i=1:grid.nkr
        for j=1:grid.nl
            k = grid.kr[i]
            l = grid.l[j]
            L[i, j, :, :] = Lop(k, l, i, j)
        end
    end
end

function populate_L!(L, grid, params, dev::GPU)
    D = @. - params.ν * grid.Krsq^(params.nν)
    
    config_kernel = @cuda launch=false Lop_kernel(L, grid.kr, grid.l, grid.nkr, grid.nl, D, params.f, params.Cg2)
    max_threads = CUDA.maxthreads(config_kernel)
    thread_size = 2^(floor(Int, log2(max_threads)/2))
    num_threads_x = min(thread_size, grid.nkr)
    num_threads_y = min(thread_size, grid.nl)
    num_blocks_x = cld(grid.nkr, num_threads_x)
    num_blocks_y = cld(grid.nl, num_threads_y)
    CUDA.@sync begin
        @cuda threads=(num_threads_x, num_threads_y) blocks=(num_blocks_x, num_blocks_y) Lop_kernel(L, grid.kr, grid.l, grid.nkr, grid.nl, D, params.f, params.Cg2)
    end
end

function Equation(params, grid)
    T = eltype(grid)
    dev = grid.device

    # For the standard diagonal problem
    # L = zeros(dev, T, (grid.nkr, grid.nl, 3))
    # L[:,:, 1, 1] .= D # for u equation
    # L[:,:, 2, 2] .= D # for v equation
    # L[:,:, 3, 3] .= D # for η equation

    # For integrating factor
    L = zeros(dev, Complex{T}, (grid.nkr, grid.nl, 3, 3))
    populate_L!(L, grid, params, dev)

    #return FourierFlows.Equation(L, NOPcalcN!, grid, dims=(grid.nkr, grid.nl, 3))
    return FourierFlows.Equation(L, calcN!, grid, dims=(grid.nkr, grid.nl, 3))
end

function set_solution!(prob, u0h, v0h, η0h)
    vars, grid, sol = prob.vars, prob.grid, prob.sol

    A = typeof(vars.uh) # determine the type of vars.uh

    sol[:,:,1] = A(u0h)
    sol[:,:,2] = A(v0h)
    sol[:,:,3] = A(η0h)

    updatevars!(prob)

    return nothing
end

function kinetic_energy(uh, vh, grid)
    return parsevalsum2(vars.uh , grid) / (2 * grid.Lx * grid.Ly) + parsevalsum2(vars.vh , grid) / (2 * grid.Lx * grid.Ly)
end
@inline kinetic_energy(prob) = kinetic_energy(prob.sol, prob.vars, prob.params, prob.grid)
@inline kinetic_energy(sol, vars, params, grid) = kinetic_energy(prob.vars.uh, prob.vars.vh, prob.grid)

function potential_energy(ηh, params, grid)
  return params.Cg2 * parsevalsum2(vars.ηh, grid) / (grid.Lx * grid.Ly)
end
@inline potential_energy(prob) = potential_energy(prob.sol, prob.vars, prob.params, prob.grid)
@inline potential_energy(sol, vars, params, grid) = potential_energy(prob.vars.ηh, prob.grid, prob.params)

@inline energy(prob) = energy(prob.sol, prob.vars, prob.params, prob.grid)
@inline energy(sol, vars, params, grid) = kinetic_energy(sol, vars, params, grid) + potential_energy(sol, vars, params, grid)
end
