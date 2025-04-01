using SpecialFunctions

matern_model(ω, p) = @. p[1] * ((ω - p[2])^2 + p[3]^2)^p[4]
log_matern_model(ω, p) = @. log(p[1]^2 * ((ω - p[2])^2 + p[3]^2)^p[4])
simple_power_law(ω, p) = @. p[2] * ω + p[1]

function matern_normalizing_constant(α, λ)
    return beta(0.5, α-0.5)/(λ^(2α-1))
end

function power_pdf(α, x_min)
    C = (α-1) / x_min
    return (ω) -> C*(ω/x_min)^(-α)
end

function matern_pdf(α, λ, η)
    C = 1/normalizing_constant(α, λ)
    return (ω) -> C/((ω-η)^2 + λ^2)^(α)
end

function log_power_likelihood(params)
    α = params[1]
    x_min = 0.1
    return N * log(α-1) - α * sum(log.(ω)) - N*(1-α)*log(x_min)
end

function log_likelihood_function(ω)
    N = length(ω)
    function log_matern_likelihood(params)
        α, λ, η = params
        C = matern_normalizing_constant(α, λ)
        return -α * sum(@. log(((ω - η)^2 + λ^2))) - N * log(C)
    end
    return log_matern_likelihood
end

function estimate_pdf(data, query_point, kernel, bandwidth)
    pointwise_contribution = sample -> kernel((sample - query_point)/bandwidth)
    return sum(pointwise_contribution, data)/length(data)/bandwidth
end

function gaussian_kernel(x)
    return 1/sqrt(2π)*exp(-x^2/2)
end

function lognormal_kernel(x)
    return 1/sqrt(2π)*exp(-x^2/2)
end

function estimate_pdf(ωs)
    μ_rel = sum(ωs) / length(ωs)
    σ_rel = sqrt(sum((ωs .- μ_rel).^2)/(length(ωs)-1))
    rel_bandwidth = 1.06 * σ_rel * length(ωs)^(-1/5) + 1e-2
    
    query_start = minimum(ωs)
    query_end = maximum(ωs)
    Npoints = length(ωs)
    
    linspace = (0:(Npoints-1))/(Npoints-1)
    #query_points = @. exp(log(query_start) + log(query_end-query_start)*linspace)
    query_points = @. query_start + (query_end - query_start)*linspace

    rel_pdf = estimate_pdf.(Ref(ωs), query_points, gaussian_kernel, rel_bandwidth)
    return query_points, rel_pdf
end

function matern_estimate(ω_i, density)
    p0 = [1.0, 1.0, 10.0, -4.0]
    matern_fit = curve_fit(log_matern_model, ω_i, log.(density), p0)
end

function power_law_estimate(ω_i, density)
    power_law_p0 = [10, -1.0]
    
    min_ω = log(minimum(ω_i))
    Δω = log(maximum(ω_i)) - min_ω
    start_ω = exp(min_ω + 0.35 * Δω)
    end_ω = exp(min_ω + 0.75 * Δω)
    
    start_index = findfirst(>=(start_ω), ω_i)
    end_index = findlast(<=(end_ω), ω_i)
    power_filter = start_index:end_index
    
    power_law_fit = curve_fit(simple_power_law, log.(ω_i[power_filter]), log.(density[power_filter]), power_law_p0)
end

function matern_ML_estimate(ω)
    lower_limits   = [0.5001, 0,   -Inf]
    initial_params = [1.3000, 1,    f0]
    upper_limits   = [Inf   , Inf,  Inf]
    
    matern_func = log_likelihood_function(ω)
    func = TwiceDifferentiable((vars) -> -matern_func(vars), initial_params);
    opt = optimize(func, lower_limits, upper_limits, initial_params)
    
    params = Optim.minimizer(opt)
    return params
end