module Collated

using JLD2
using Printf

export 
    CollatedInput,
    map_input,
    CollatedOutput,
    write!,
    close!

mutable struct CollatedOutput
    output_file :: JLD2.JLDFile
    line_limit :: Int
    line_index :: Int
    filename :: String
    file_index :: Int
end

mutable struct CollatedInput
    directory :: String
    base_prefix :: String
end

function CollatedOutput(directory::String, base_filename::String, line_limit::Int)::CollatedOutput
    if !isdir(directory); mkdir(directory); end
    if endswith(base_filename, ".out")
        base_filename = base_filename[1:end-4]
    end
    return CollatedOutput(directory * "/" * base_filename, line_limit)
end

function CollatedOutput(filename::String, line_limit::Int)::CollatedOutput
    output_file = open!(get_filename(filename, 0))
    output = CollatedOutput(output_file, line_limit, 0, filename, 0)
    return output
end

function write!(output::CollatedOutput, key::String, value::Any)
    output.output_file[key] = value
    output.line_index += 1
    if (output.line_index >= output.line_limit)
        open_next_file(output)
    end
    return nothing
end

function open_next_file(output::CollatedOutput)
    close!(output)
    output.line_index = 0
    output.file_index += 1
    output_filename = get_filename(output)
    output.output_file = open!(output_filename)
    return nothing
end

function get_filename(filename::String, file_index::Int)::String
    return @sprintf("%s_%08d.out", filename, file_index)
end

function get_filename(output::CollatedOutput)::String
    return get_filename(output.filename, output.file_index)
end

function open!(filename::String)::JLD2.JLDFile
    return jldopen(filename, "w")
end

function close!(output::CollatedOutput)::Nothing
    close(output.output_file)
end

function map_input(input::CollatedInput, func; group_key=""::String)
    index = 0
    filename(idx) = get_filename(input.directory * "/" * input.base_prefix, idx)
    results = []
    while isfile(filename(index))
        file = jldopen(filename(index))
        elements = []
        if(group_key=="")
            elements = keys(file)
        else
            elements = keys(file[group_key])
        end
        for key=elements
            push!(results, func(file[key]))
        end
        close(file)
        index += 1
    end
    return results
end
end