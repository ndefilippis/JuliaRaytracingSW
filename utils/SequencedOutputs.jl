module SequencedOutputs
using FourierFlows
using JLD2

export SequencedOutput, saveoutput, saveproblem, close, setindex!

mutable struct SequencedOutput{T}
    max_writes :: Int
    current_writes :: Int
    file_index :: Int
    get_filename
    output_file :: T
end

function SequencedOutput(prob, filename_function, fields, max_writes)
     filename = filename_function(0)
     output_file = Output(prob, filename, fields)
     return SequencedOutput(max_writes, 0, 0, filename_function, output_file)
end

function SequencedOutput(filename_function, max_writes)
    filename = filename_function(0)
    output_file = jldopen(filename, "w")
    return SequencedOutput(max_writes, 0, 0, filename_function, output_file)
end

function create_new_output(output::SequencedOutput{Output})
    filename = output.get_filename(output.file_index)
    return Output(output.output_file.prob, filename, output.output_file.fields)
end

function create_new_output(output::SequencedOutput{JLD2.JLDFile{T}}) where {T}
    filename = output.get_filename(output.file_index)
    return jldopen(filename, "w")
end

function check_writes(output::SequencedOutput)
    if output.current_writes >= output.max_writes
        close(output)
        output.current_writes = 0
        output.file_index += 1
        output.output_file = create_new_output(output)
    end
end

function saveproblem(output::SequencedOutput{Output})
    FourierFlows.saveproblem(output.output_file)
    output.current_writes += 1
    check_writes(output)
end

function saveoutput(output::SequencedOutput{Output})
    FourierFlows.saveoutput(output.output_file)
    output.current_writes += length(output.output_file.fields)
    check_writes(output)
end

function Base.setindex!(output::SequencedOutput{JLD2.JLDFile{T}}, obj, name::AbstractString) where {T}
    output.output_file[name] = obj
    output.current_writes += 1
    check_writes(output)
    return nothing
end

function close(output::SequencedOutput{Output})
    return nothing
end

function close(output::SequencedOutput{JLD2.JLDFile{T}}) where {T}
    JLD2.close(output.output_file)
end

end
