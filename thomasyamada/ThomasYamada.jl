module ThomasYamada
export
  Problem,
  set_solution!,
  updatevars!,

  barotropic_energy,
  baroclinic_energy,
  parsevalsum2

using FourierFlows
using FourierFlows: parsevalsum2

using LinearAlgebra: mul!, ldiv!

struct Params{T} <: AbstractParams
   ν :: T         # Hyperviscosity coefficient
  nν :: Int       # Order of the hyperviscous operator
  Ro :: T         # Rossby numer
end

struct Vars{Aphys, Atrans} <: AbstractVars
    uc  :: Aphys
    vc  :: Aphys
    ut  :: Aphys
    vt  :: Aphys
    ζt  :: Aphys
    ψt  :: Aphys
    pc  :: Aphys
    qc :: Aphys
    uch  :: Atrans
    vch  :: Atrans
    uth  :: Atrans
    vth  :: Atrans
    ζth  :: Atrans
    ψth  :: Atrans
    pch  :: Atrans
    qch :: Atrans
end

function Vars(grid)
  Dev = typeof(grid.device)
  T = eltype(grid)

  @devzeros Dev T (grid.nx, grid.ny) uc vc ut vt ζt ψt pc qc
  @devzeros Dev Complex{T} (grid.nkr, grid.nl) uch vch uth vth ζth ψth pch qch

  return Vars(uc, vc, ut, vt, ζt, ψt, pc, qc, uch, vch, uth, vth, ζth, ψth, pch, qch)
end

function Problem(dev::Device = CPU();
    nx = 128,
    ny = nx,
    Lx = 2π,
    Ly = Lx,
    ν  = 3.5e-25,
    nν = 8,
    Ro = 0.2,
    stepper = "ETDRK4",
    dt = 5e-2,
    aliased_fraction = 1/3,
    T = Float64)
   
    grid = TwoDGrid(dev; nx, Lx, ny, Ly, aliased_fraction, T)
    params = Params{T}(ν, nν, Ro)
    vars = Vars(grid)
    equation = Equation(params, grid)
    
    prob = FourierFlows.Problem(equation, stepper, dt, grid, vars, params)
end

function updatevars!(prob)
    vars, grid, sol = prob.vars, prob.grid, prob.sol

    dealias!(sol, grid)
    @. vars.ζth = sol[:,:,1]
    @. vars.uch = sol[:,:,2]
    @. vars.vch = sol[:,:,3]
    @. vars.pch = sol[:,:,4]

    ldiv!(vars.ζt, grid.rfftplan, deepcopy(sol[:,:,1])) # use deepcopy() because irfft destroys its input
    ldiv!(vars.uc, grid.rfftplan, deepcopy(sol[:,:,2])) # use deepcopy() because irfft destroys its input
    ldiv!(vars.vc, grid.rfftplan, deepcopy(sol[:,:,3])) # use deepcopy() because irfft destroys its input
    ldiv!(vars.pc, grid.rfftplan, deepcopy(sol[:,:,4])) # use deepcopy() because irfft destroys its input
        
    streamfunctionfromvorticity!(vars.ψth, vars.ζth, grid)
    
    @. vars.uth = -im * grid.l  * vars.ψth
    @. vars.vth =  im * grid.kr * vars.ψth 
    @. vars.qch =  im * grid.kr * vars.vch - im * grid.l * vars.uch - vars.pch
    
    ldiv!(vars.ut, grid.rfftplan, deepcopy(vars.uth)) # use deepcopy() because irfft destroys its input
    ldiv!(vars.vt, grid.rfftplan, deepcopy(vars.vth)) # use deepcopy() because irfft destroys its input
    ldiv!(vars.qc, grid.rfftplan, deepcopy(vars.qch)) # use deepcopy() because irfft destroys its input
    
    return nothing
end

function streamfunctionfromvorticity!(ψh, ζh, grid)
    @. ψh = - ζh * grid.invKrsq
end

function calcN!(N, sol, t, clock, vars, params, grid)
    dealias!(sol, grid)
    
    @. vars.ζth = @view sol[:,:,1]
    @. vars.uch = @view sol[:,:,2]
    @. vars.vch = @view sol[:,:,3]
    @. vars.pch = @view sol[:,:,4]
    
    streamfunctionfromvorticity!(vars.ψth, vars.ζth, grid)
    
    @. vars.uth = -im * grid.l  * vars.ψth
    @. vars.vth =  im * grid.kr * vars.ψth 

    # Calculate linear dynamics    
    @. N[:,:, 1] =   0.
    @. N[:,:, 2] =   vars.vch - im * grid.kr * vars.pch
    @. N[:,:, 3] = - vars.uch - im * grid.l  * vars.pch
    @. N[:,:, 4] = - im * grid.kr * vars.uch - im * grid.l * vars.vch
    
    # Use pch and qch as scratch variables because ldiv! destroys its input
    @. vars.pch = vars.ζth
    ldiv!(vars.ζt, grid.rfftplan, vars.pch)
    @. vars.pch = vars.uth
    @. vars.qch = vars.vth
    ldiv!(vars.ut, grid.rfftplan, vars.pch)
    ldiv!(vars.vt, grid.rfftplan, vars.qch)
    @. vars.pch = vars.uch
    @. vars.qch = vars.vch
    ldiv!(vars.uc, grid.rfftplan, vars.pch)
    ldiv!(vars.vc, grid.rfftplan, vars.qch)
    
    calcN_vorticity!(N, sol, t, clock, vars, params, grid)
    calcN_baroclinic!(N, sol, t, clock, vars, params, grid)
    
    @. vars.pch = @view sol[:,:,4] # Because we used pch as a scratch variable
    calcN_pressure!(N, sol, t, clock, vars, params, grid)
    return nothing
end


function calcN_vorticity!(N, sol, t, clock, vars, params, grid)
    # =====
    # Compute ζt Rossby order terms
    # =====
    ζtN = @views N[:,:,1]
    
    # Calculate jacobian term
    # ∂(ζt)/∂t = -Ro*(J(ψ, ζ) + (∂²x - ∂²y)(uc*vc) + ∂²xy(vc^2) - ∂²xy(uc^2)
    # Note that J(f, g) = f_x*g_y - f_y*g_x = ∂y(f_xg) - ∂x(f_yg)
    vζ = vars.qc          # Use qc as a scratch variable
    @. vζ = vars.vt * vars.ζt
    vζh = vars.qch
    mul!(vζh, grid.rfftplan, vζ)

    uζ = vars.pc            # Use pc as a scratch variable
    @. uζ = vars.ut * vars.ζt
    uζh = vars.pch
    mul!(uζh, grid.rfftplan, uζ)
    @. ζtN += -params.Ro * (im * grid.l * vζh + im * grid.kr * uζh)
    
    # Calculate baroclinic terms
    uv = vars.qc
    @. uv = vars.uc * vars.vc
    uvh = vars.qch
    mul!(uvh, grid.rfftplan, uv)
    @. ζtN += -params.Ro * (- grid.kr^2 + grid.l^2) * uvh
    
    v2 = vars.qc # Use qc and pc as scratch variables
    u2 = vars.pc
    @. v2 = vars.vc * vars.vc
    @. u2 = vars.uc * vars.uc
    v2h = vars.qch
    u2h = vars.pch
    mul!(v2h, grid.rfftplan, v2)
    mul!(u2h, grid.rfftplan, u2)
    @. ζtN += -params.Ro * (- grid.kr * grid.l * v2h + grid.kr * grid.l * u2h)
end

function calcN_baroclinic!(N, sol, t, clock, vars, params, grid)
    # =====
    # Compute the uc and vc equation Rossby order terms
    # =====
    ucN = @views N[:,:,2]
    vcN = @views N[:,:,3]
    
    ucut = vars.qc # Use qc and pc as a scratch variable
    vcvt = vars.pc
    @. ucut = vars.ut * vars.uc
    @. vcvt = vars.vt * vars.vc
    ucuth = vars.qch
    vcvth = vars.pch
    mul!(ucuth, grid.rfftplan, ucut)
    mul!(vcvth, grid.rfftplan, vcvt)
    @. ucN += -params.Ro * (im * grid.kr * (ucuth))
    @. vcN += -params.Ro * (im * grid.l  * (vcvth))
    
    vtucy = vars.qc
    vcuty = vars.pc
    vtucyh = vars.qch
    vcutyh = vars.pch
    @. vtucyh = im * grid.l * vars.uch
    @. vcutyh = im * grid.l * vars.uth
    ldiv!(vtucy, grid.rfftplan, vtucyh)
    ldiv!(vcuty, grid.rfftplan, vcutyh)
    @. vtucy *= vars.vt 
    @. vcuty *= vars.vc
    mul!(vtucyh, grid.rfftplan, vtucy)
    mul!(vcutyh, grid.rfftplan, vcuty)
    @. ucN += -params.Ro * (vtucyh + vcutyh)
    
    utvcx = vars.qc
    ucvtx = vars.pc
    utvcxh = vars.qch
    ucvtxh = vars.pch
    @. utvcxh = im * grid.kr * vars.vch
    @. ucvtxh = im * grid.kr * vars.vth
    ldiv!(utvcx, grid.rfftplan, utvcxh)
    ldiv!(ucvtx, grid.rfftplan, ucvtxh)
    @. utvcx *= vars.ut 
    @. ucvtx *= vars.uc
    mul!(utvcxh, grid.rfftplan, utvcx)
    mul!(ucvtxh, grid.rfftplan, ucvtx)
    @. vcN += -params.Ro * (utvcxh + ucvtxh)    
end

function calcN_pressure!(N, sol, t, clock, vars, params, grid)
    # =====
    # Compute the pc equation Rossby terms
    # =====
    pcN = @views N[:,:,4]
    
    utpcxh = vars.uch # Use uch and vch as scratch variables
    vtpcyh = vars.vch
    @. utpcxh = im * grid.kr * vars.pch
    @. vtpcyh = im * grid.l  * vars.pch
    
    utpcx = vars.uc
    vtpcy = vars.vc
    ldiv!(utpcx, grid.rfftplan, utpcxh)
    ldiv!(vtpcy, grid.rfftplan, vtpcyh)
    @. utpcx *= vars.ut
    @. vtpcy *= vars.vt
    mul!(utpcxh, grid.rfftplan, utpcx)
    mul!(vtpcyh, grid.rfftplan, vtpcy)
    @. pcN += -params.Ro * (utpcxh + vtpcyh)
end


function Equation(params, grid)
    T = eltype(grid)
    dev = grid.device

    L = zeros(dev, T, (grid.nkr, grid.nl, 4))
    D = @. - params.ν * grid.kr^(2*params.nν)

    L[:,:, 1] .= D # for ζt equation
    L[:,:, 2] .= D # for uc equation
    L[:,:, 3] .= D # for vc equation
    L[:,:, 4] .= D # for pc equation
    
    return FourierFlows.Equation(L, calcN!, grid)
end

function set_solution!(prob, ζ0h, u0h, v0h, p0h)
    vars, grid, sol = prob.vars, prob.grid, prob.sol

    A = typeof(vars.ζth) # determine the type of vars.u
    
    #@. vars.ζth = A(ζ0h)
    #@. vars.uch = A(u0h)
    #@. vars.vch = A(v0h)
    #@. vars.pch = A(p0h)

    # below, e.g., A(u0) converts u0 to the same type as vars expects
    # (useful when u0 is a CPU array but grid.device is GPU)
    # mul!(vars.ζth, grid.rfftplan, A(ζ0))
    # mul!(vars.uch, grid.rfftplan, A(u0))
    # mul!(vars.vch, grid.rfftplan, A(v0))
    # mul!(vars.pch, grid.rfftplan, A(p0))

    sol[:,:, 1] = A(ζ0h)
    sol[:,:, 2] = A(u0h)
    sol[:,:, 3] = A(v0h)
    sol[:,:, 4] = A(p0h)

    updatevars!(prob)

    return nothing
end

#function parsevalsum2(uh, grid)
#  if size(uh, 1) == grid.nkr  # uh is in conjugate symmetric form
#    Σ = sum(abs2, uh[1, :])                  # k = 0 modes
#    Σ += sum(abs2, uh[grid.nkr, :])          # k = nx/2 modes
#    Σ += 2 * sum(abs2, uh[2:grid.nkr-1, :])  # sum twice for 0 < k < nx/2 modes
#  else # count every mode once
#    Σ = sum(abs2, uh)
#  end
#
#  Σ *= grid.Lx * grid.Ly / (grid.nx^2 * grid.ny^2) # normalization for dft
#
#  return Σ
#end

function baroclinic_energy(prob)
    uch = @views prob.sol[:,:,2]
    vch = @views prob.sol[:,:,3]
    pch = @views prob.sol[:,:,4]
    return parsevalsum2(uch, prob.grid) + parsevalsum2(vch, prob.grid) + parsevalsum2(pch, prob.grid)
end

function barotropic_energy(prob)
    ζth = @views prob.sol[:,:,1]
    return parsevalsum2(sqrt.(prob.grid.invKrsq) .* ζth, prob.grid)
end
end
