module TwoLayerQG
export 
    Problem,
    set_solution!,
    enforce_reality_condition!,
    updatevars!,
    energy,
    kinetic_energy,
    potential_energy,
    enstrophy

using Revise
using FourierFlows
using FourierFlows: parsevalsum2, parsevalsum, plan_flows_rfft
using CUDA
using StaticArrays
using KernelAbstractions
using LinearAlgebra: mul!, ldiv!

using ..IFMAB3

# Only implemented for equal layer heights

struct Params{T, Trfft} <: AbstractParams
    U :: T         # Background flow velocity
    μ :: T         # Linear bottom drag coefficient
    ν :: T         # Hyperviscosity coefficient
    nν :: Int       # Order of the hyperviscous operator
    F :: T         # Function of Rossby deformation wavenumber
    rfftplan :: Trfft
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
    U = 0.5,
    μ = 1e-2,
    ν = 1e-6,
    nν = 4,
    f0 = 3.0,
    Cg = 1.0,
    δρρ0 = 0.2,
    stepper = "IFMAB3",
    dt = 5e-2,
    aliased_fraction = 1/3,
    T = Float32,
    use_filter=false,
    stepper_kwargs...)

    A = device_array(dev)
    grid = TwoDGrid(dev; nx, Lx, ny, Ly, aliased_fraction, T)
    
    rfftplanlayered = plan_flows_rfft(A{T, 3}(undef, grid.nx, grid.ny, 2), [1, 2])
    
    params = Params(T(U), T(μ), T(ν), nν, T(2*f0^2/Cg^2/δρρ0), rfftplanlayered)
    vars = Vars(grid)
    equation = Equation(params, grid)
    if stepper == "IFMAB3"
        clock = FourierFlows.Clock{T}(dt, 0, 0)
        timestepper = IFMAB3TimeStepper(equation, dt, dev; diagonal=false, use_filter, stepper_kwargs...)
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
    @. q1h = -grid.Krsq * ψ1h + params.F * (ψ2h - ψ1h)
    @. q2h = -grid.Krsq * ψ2h + params.F * (ψ1h - ψ2h)
end

function streamfunctionfrompv!(ψh, qh, grid, params)
    ψ1h = @view ψh[:,:,1]
    ψ2h = @view ψh[:,:,2]
    q1h = @view qh[:,:,1]
    q2h = @view qh[:,:,2]
    
    @. ψ1h = -(grid.Krsq * q1h + params.F * (q1h + q2h))
    @. ψ2h = -(grid.Krsq * q2h + params.F * (q1h + q2h))
    @. ψh /= grid.Krsq + 2*params.F
    @. ψh *= grid.invKrsq
end

function updatevars!(prob)
    vars, grid, sol, params = prob.vars, prob.grid, prob.sol, prob.params

    dealias!(sol, grid)
    @. vars.qh = sol
    streamfunctionfrompv!(vars.ψh, vars.qh, grid, params)
    @. vars.ζh = -grid.Krsq * vars.ψh
    @. vars.uh = -1im * grid.l  * vars.ψh
    @. vars.vh =  1im * grid.kr * vars.ψh

    invtransform!(vars.q, deepcopy(vars.qh), params) # use deepcopy() because irfft destroys its input
    invtransform!(vars.ψ, deepcopy(vars.ψh), params) # use deepcopy() because irfft destroys its input
    invtransform!(vars.ζ, deepcopy(vars.ζh), params)
    invtransform!(vars.u, deepcopy(vars.uh), params)
    invtransform!(vars.v, deepcopy(vars.vh), params)
    return nothing
end

@inline function fwdtransform!(varh, var, params)
    mul!(varh, params.rfftplan, var)
end

@inline function invtransform!(var, varh, params)
    ldiv!(var, params.rfftplan, varh)
end

function enforce_reality_condition!(prob)
    vars, grid, sol = prob.vars, prob.grid, prob.sol

    dealias!(sol, grid)
    @. vars.qh = sol

    updatevars!(prob)
        
    fwdtransform!(vars.qh, deepcopy(vars.q), prob.params)
    fwdtransform!(vars.ψh, deepcopy(vars.ψ), prob.params)
    return nothing
end

function calcN!(N, sol, t, clock, vars, params, grid)
    dealias!(sol, grid)
    
    @. vars.qh = sol
    streamfunctionfrompv!(vars.ψh, vars.qh, grid, params)
    qh = vars.ζh
    @. qh = vars.qh
    invtransform!(vars.q, qh, params)

    # Use ζ and ζh as scratch variables
    # Calculate advective terms
    # q_t = -J(ψ, q)
    # Use the fact that J(f, g) = (f_xg)_y - (f_yg)_x
    ψxqh = vars.ζh
    ψxq = vars.ζ
    @. ψxqh = 1im * grid.kr * vars.ψh
    invtransform!(ψxq, ψxqh, params)
    @. ψxq *= vars.q
    fwdtransform!(ψxqh, ψxq, params)
    @. N = -1im * grid.l * ψxqh

    ψyqh = vars.ζh
    ψyq = vars.ζ
    @. ψyqh = 1im * grid.l * vars.ψh
    invtransform!(ψyq, ψyqh, params)
    @. ψyq *= vars.q
    fwdtransform!(ψyqh, ψyq, params)
    @. N += 1im * grid.kr * ψyqh

    return nothing
end

@kernel function L_kernel!(L, k, l, F, U, μ, D)
    i, j = @index(Global, NTuple)   
    K2 = k[i]^2 + l[j]^2
    K2inv = K2 == 0 ? 0 : 1.0/K2
    
    PV_term = SVector{2,Complex{Float32}}(-2im*k[i]*F*U, 2im*k[i]*F*U)
    drag_term = SVector{2,Complex{Float32}}(0, μ*K2)
    psi_terms = PV_term + drag_term
            
    Sinv = SMatrix{2,2,Complex{Float32}}(-K2-F, -F, -F, -K2-F)/(K2+2*F)*K2inv
    
    @inbounds @view(L[i, j, :, :]) .= psi_terms .* Sinv
    L[i, j, 1, 1] += -1im*k[i]*U + D[i,j]
    L[i, j, 2, 2] +=  1im*k[i]*U + D[i,j]
end

function populate_L!(L, grid, params)
    D = @. - params.ν * grid.Krsq^(params.nν)

    backend = KernelAbstractions.get_backend(L)
    kernel! = L_kernel!(backend)
    kernel!(L, grid.kr, grid.l, params.F, params.U, params.μ, D, ndrange=size(grid.Krsq))
end

function Equation(params, grid)
    T = eltype(grid)
    dev = grid.device

    L = zeros(dev, Complex{T}, (grid.nkr, grid.nl, 2, 2))
    backend = get_backend(L)
    populate_L!(L, grid, params)
    KernelAbstractions.synchronize(backend)
    
    return FourierFlows.Equation(L, calcN!, grid, dims=(grid.nkr, grid.nl, 2))
end

function set_solution!(prob, q0h)
    vars, grid, sol = prob.vars, prob.grid, prob.sol

    A = typeof(vars.qh) # determine the type of vars.uh
    sol .= A(q0h)
    updatevars!(prob)

    return nothing
end

function kinetic_energy(vars, params, grid, sol)
  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, grid, params)

  abs²∇𝐮h = vars.uh        # use vars.uh as scratch variable
  @. abs²∇𝐮h = grid.Krsq * abs2(vars.ψh)

  KE_1 = parsevalsum(@views(abs²∇𝐮h[:,:,1]) , grid) / (grid.Lx * grid.Ly) # factor 2 cancels out via H/2
  KE_2 = parsevalsum(@views(abs²∇𝐮h[:,:,2]) , grid) / (grid.Lx * grid.Ly)
  return (KE_1, KE_2)
end

@inline kinetic_energy(prob) = kinetic_energy(prob.vars, prob.params, prob.grid, prob.sol)

function potential_energy(vars, params, grid, sol)
  @. vars.qh = sol
  streamfunctionfrompv!(vars.ψh, vars.qh, grid, params)
    
  PE = 1 / (2 * grid.Lx * grid.Ly) * params.F * parsevalsum(abs2.(view(vars.ψh, :, :, 1) .- view(vars.ψh, :, :, 2)), grid)
  return PE
end

@inline potential_energy(prob) = potential_energy(prob.vars, prob.params, prob.grid, prob.sol)
end