using GeophysicalFlows
using Printf

function set_up_problem(filename)
    L = 2π
    jldopen(filename) do ic_file
        ψh = ic_file["ic/ψh"]
        @unpack g, f₀, β, ρ, H, U, μ = ic_file["params"]
        nlayers = 2
        dev = CPU()
        L = 2π
        nx = size(ψh, 2)
        U = U[1,1,:]
        ρ = [ρ[1], ρ[2]]
        prob = MultiLayerQG.Problem(nlayers, dev; nx, Lx=L, f₀, g, H, ρ, U, μ, β, aliased_fraction=0)
        pvfromstreamfunction!(prob.sol, ψh, prob.params, prob.grid)
        MultiLayerQG.updatevars!(prob)
        return prob
    end
end

function display_energetics(prob)
    KE, PE = MultiLayerQG.energies(prob)
    KE₁, KE₂ = KE
    
    U = abs(prob.params.U[1])
    λ = 1/sqrt(prob.params.f₀^2/prob.params.g′*(prob.params.H[1] + prob.params.H[2])/(prob.params.H[1]*prob.params.H[2]))
    κ = prob.params.μ
    nondim_κ = κ*U/λ
    nondim_ℓ = 3.2 * exp( 0.36 / nondim_κ )
    ℓ = nondim_ℓ * λ
    V = U*ℓ/λ
    println("Parameters:")
    println(@sprintf("λ:%15.5f", λ))
    println(@sprintf("κ*:%14.5f", nondim_κ))
    println(@sprintf("ℓ*:%14.5f", nondim_ℓ))
    println(@sprintf("V:%15.5f", V))
    println("=================")
    println(@sprintf("pred KE:%14.5f", V^2))
    println(@sprintf("real top KE:%10.5f", KE₁))
    println(@sprintf("real bot KE:%10.5f", KE₂))
    println(@sprintf("real tot KE:%10.5f", KE₁ + KE₂))
end

struct CollatedOutput
    output_file
    line_limit :: Int
    line_index :: Int
    file_pattern :: String
    file_index :: Int
end

function write_jld2_line(output::CollatedOutput, key::String, value::Any)
    output.output_file[key] = value
    output.line_index += 1
    if (output.line_index >= output.line_limit)
        
    end
end