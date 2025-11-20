module IFMAB3
export IFMAB3TimeStepper, stepforward!

using FourierFlows, CUDA, StaticArrays

import FourierFlows: makefilter
import FourierFlows: AbstractTimeStepper

const ab3h1 = 23/12
const ab3h2 = 16/12
const ab3h3 = 5/12

struct IFMRK4TimeStepper{T,TL,Tf} <: FourierFlows.AbstractTimeStepper{T}
  # IFMAB3 coefficents
     expLdt  :: TL
     expL½dt :: TL
         k₁ :: T
         k₂ :: T
         k₃ :: T
         k₄ :: T
    filter   :: Tf
end


function getexpLs(dt, equation, dev::CPU)
    expLdt = mapslices(exp, equation.L * dt, dims=(3, 4))
    expL½dt = mapslices(exp, equation.L * 0.5 * dt, dims=(3, 4))
    return expLdt, exp½Ldt
end

function kernel_exp(result, A, dt, Nx, Ny, ::Val{N}) where N
    i = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    j = blockDim().y * (blockIdx().y - 1) + threadIdx().y
    if i > Nx || j > Ny
        return
    end
    @inbounds Lkj = SMatrix{N, N, Complex{Float64}}(@view(A[i, j, :, :]))
    @inbounds @view(result[i, j, :, :]) .= CUDA.exp(dt * Lkj)
    return
end


function getexpLs(dt, equation, dev::GPU)
    Nx = size(equation.L, 1)
    Ny = size(equation.L, 2)
    
    expLdt = similar(equation.L)
    exp½Ldt = similar(equation.L)

    eq_dims = Val(size(equation.L, 3))

    config_kernel = @cuda launch=false kernel_exp(expLdt, equation.L, dt, Nx, Ny, eq_dims)
    max_threads = CUDA.maxthreads(config_kernel)
    
    thread_size = 2^(floor(Int, log2(max_threads)/2))
    num_threads_x = min(thread_size, Nx)
    num_threads_y = min(thread_size, Ny)
    num_blocks_x = cld(Nx, num_threads_x)
    num_blocks_y = cld(Ny, num_threads_y)
    CUDA.@sync begin
        @cuda threads=(num_threads_x, num_threads_y) blocks=(num_blocks_x, num_blocks_y) kernel_exp(expLdt, equation.L, dt, Nx, Ny, eq_dims)
        @cuda threads=(num_threads_x, num_threads_y) blocks=(num_blocks_x, num_blocks_y) kernel_exp(exp½Ldt, equation.L, 0.5*dt, Nx, Ny, eq_dims)
    end
    return expLdt, exp½Ldt
end

function IFMAB3TimeStepper(equation::FourierFlows.Equation, dt, dev::Device=CPU(); diagonal=false, use_filter=true, filter_kwargs...)
    dt = fltype(equation.T)(dt) # ensure dt is correct type.
    expLdt = nothing
    exp½Ldt = nothing
    if diagonal
        expLdt = exp.(equation.L * dt)
        exp½Ldt = exp.(0.5 * equation.L * dt)
    else
        expLdt, exp½Ldt = getexpLs(dt, equation, dev)
    end
    @devzeros typeof(dev) equation.T equation.dims k₁ k₂ k₃ k₄ N

    if use_filter
        filter = makefilter(equation; filter_kwargs...)
    else
        @devzeros typeof(dev) equation.T equation.dims filter
        @. filter = 1
    end
  
    return IFMAB3TimeStepper(expLdt, exp½Ldt, k₁, k₂, k₃, k₄, filter)
end

function mv_mul_kernel(y, A, x, Nx, Ny, ::Val{N}) where N
    i = blockDim().x * (blockIdx().x - 1) + threadIdx().x
    j = blockDim().y * (blockIdx().y - 1) + threadIdx().y
    if i > Nx || j > Ny
        return
    end
    @inbounds Lkj = SMatrix{N, N}(@view(A[i, j, :, :]))
    @inbounds xkj = SVector{N}(@view(x[i, j, :]))
    @inbounds @view(y[i, j, :]) .= Lkj * xkj
    return
end

function mvmul!(y, A::AbstractArray{T, 2}, x) where {T}
    @. y = A * x
end

function mvmul!(y, A::AbstractArray{T, 3}, x) where {T}
    @. y = A * x
end

function mvmul!(y, A::CuArray{T, 4}, x) where {T}
    Nx = size(x, 1)
    Ny = size(x, 2)
    eq_dims = Val{size(x,3)}()
    
    config_kernel = @cuda launch=false mv_mul_kernel(y, A, x, Nx, Ny, eq_dims)
    max_threads = CUDA.maxthreads(config_kernel)
    thread_size = 2^(floor(Int, log2(max_threads)/2))
    num_threads_x = min(thread_size, Nx)
    num_threads_y = min(thread_size, Ny)
    num_blocks_x = cld(Nx, num_threads_x)
    num_blocks_y = cld(Ny, num_threads_y)
    @cuda threads=(num_threads_x, num_threads_y) blocks=(num_blocks_x, num_blocks_y) mv_mul_kernel(y, A, x, Nx, Ny, eq_dims)
end

function mvmul!(y, A::Array{T, 4}, x) where {T}
    y .= dropdims(sum(permutedims(A, [1, 2, 4, 3]) .* x, dims=3), dims=3)
end

function IFMAB3update!(sol::Array, ts, clock)
    if clock.step < 3  # forward Euler steps to initialize AB3
        @. sol += clock.dt * ts.N    # Update
        mvmul!(sol, ts.expLdt, sol)
    else   # Otherwise, stepforward with 3rd order Adams Bashforth:
        mvmul!(ts.expLdtN₋₁,  ts.expLdt,  ts.N₋₁)
        mvmul!(ts.exp2LdtN₋₂, ts.exp2Ldt, ts.N₋₂)
        @. sol += clock.dt * (ab3h1 * ts.N - ab3h2 * ts.expLdtN₋₁ + ab3h3 * ts.exp2LdtN₋₂)
        mvmul!(sol, ts.expLdt, sol)
    end
    return nothing
end

function IFMAB3update!(sol::CuArray, ts, clock)
    if clock.step < 3  # forward Euler steps to initialize AB3
        @. sol += clock.dt * ts.N    # Update
        CUDA.@sync mvmul!(sol, ts.expLdt, sol)
    else   # Otherwise, stepforward with 3rd order Adams Bashforth:
        CUDA.@sync begin
            mvmul!(ts.expLdtN₋₁,  ts.expLdt,  ts.N₋₁)
            mvmul!(ts.exp2LdtN₋₂, ts.exp2Ldt, ts.N₋₂)
        end
        @. sol += clock.dt * (ab3h1 * ts.N - ab3h2 * ts.expLdtN₋₁ + ab3h3 * ts.exp2LdtN₋₂)
        CUDA.@sync mvmul!(sol, ts.expLdt, sol)
    end
    return nothing
end

function FourierFlows.stepforward!(sol, clock, ts::IFMAB3TimeStepper, equation, vars, params, grid)
  equation.calcN!(ts.N₁, sol, clock.t, clock, vars, params, grid)

  equation.calcN!(ts.N₂, sol, clock.t, clock, vars, params, grid)

  equation.calcN!(ts.N₃, sol, clock.t, clock, vars, params, grid)
    
  equation.calcN!(ts.N₄, sol, clock.t, clock, vars, params, grid)
  IFMAB3update!(sol, ts, clock)
  @. sol = ts.filter * sol

  clock.t += clock.dt
  clock.step += 1

  @. ts.N₋₂ = ts.N₋₁          # Store
  @. ts.N₋₁ = ts.N            # ... previous values of N

  return nothing
end
end