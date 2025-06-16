abstract type RaytracingModel end

function set_initial_condition!(::RaytracingModel, prob, grid, dev)
    throw("set_initial_condition! not defined")
    return
end

function create_fourier_flows_problem(::RaytracingModel, common_params)
    throw("create_fourier_flows_problem! not defined")
    return
end

function set_initial_condition!(::RaytracingModel, prob, grid, dev)
    throw("set_initial_condition! not defined")
    return
end

function get_streamfunction!(ψh, ::RaytracingModel, prob)
    throw("get_streamfunction! not defined")
    return
end

function estimate_max_U(::RaytracingModel)
    throw("estimate_max_U not defined")
    return
end