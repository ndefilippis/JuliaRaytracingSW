using JLD2;

function load_packet_analysis_file(filename, load_velocity=false)
    packet_group = "p/"
    times = 0;
    x = 0;
    k = 0;
    u = 0;
	jldopen(filename, "r") do file
		N = length(keys(file[packet_group * "t"]));
		Npackets = length(keys(file[packet_group * "x"]));

		times = zeros(N, 1);
		x = zeros(N, Npackets, 2);
		k = zeros(N, Npackets, 2);
        u = zeros(N, Npackets, 2);
		index = 1;
		for ts in keys(file[packet_group * "t"])
			times[index] = file[packet_group * "t/$ts"];
			packet_index = 1;
			for packet in keys(file[packet_group * "x"])
				x[index, packet_index, :] = file[packet_group * "x/$packet/$ts"];
				k[index, packet_index, :] = file[packet_group * "k/$packet/$ts"];
                if(load_velocity)
                    u[index, packet_index, :] = file[packet_group * "u/$packet/$ts"];
                end
				packet_index = packet_index + 1;
			end
			index = index + 1;
		end
	end
    if(load_velocity)
        return times, x, k, u;
    else
        return times, x, k;
    end
end

function load_packet_analysis_files(filename_function, indices, load_velocity=false)
    packet_group = "p/"
    times = 0;
    x = 0;
    k = 0;
    u = 0;
    total_N = 0;
    Npackets = 0;
    for idx=indices
        filename = filename_function(idx)
        jldopen(filename, "r") do file
            total_N += length(keys(file[packet_group * "t"]));
            Npackets = length(keys(file[packet_group * "x"]));
        end
    end
    times = zeros(total_N);
    x = zeros(total_N, Npackets, 2);
    k = zeros(total_N, Npackets, 2);
    u = zeros(total_N, Npackets, 2);
    base_index = 0
    for idx=indices
        filename = filename_function(idx)
    	println("Reading file: " *filename)
        jldopen(filename, "r") do file
    		N = length(keys(file[packet_group * "t"]));
    		Npackets = length(keys(file[packet_group * "x"]));
    		index = 1;
    		for ts in keys(file[packet_group * "t"])
    			times[base_index + index] = file[packet_group * "t/$ts"];
    			packet_index = 1;
    			for packet in keys(file[packet_group * "x"])
    				x[base_index + index, packet_index, :] = file[packet_group * "x/$packet/$ts"];
    				k[base_index + index, packet_index, :] = file[packet_group * "k/$packet/$ts"];
                    if(load_velocity)
                        u[base_index + index, packet_index, :] = file[packet_group * "u/$packet/$ts"];
                    end
    				packet_index = packet_index + 1;
    			end
    			index += 1;
    		end
        	base_index += N
        end
    end
    if(load_velocity)
        return times, x, k, u;
    else
        return times, x, k;
    end
end

function load_packet_analysis_files_collated(filename_function, indices; packet_idxs=nothing, load_velocity=false)
    packet_group = "p/"
    total_N = 0;
    Npackets = 0;
    for idx=indices
        filename = filename_function(idx)
        jldopen(filename, "r") do file
            total_N += length(keys(file[packet_group * "t"]));
            first_key = keys(file[packet_group * "x"])[1]
            Npackets = size(file[packet_group * "x/" * first_key], 1);
        end
    end
    if packet_idxs == nothing
        packet_idxs = 1:Npackets
    else
        Npackets = length(packet_idxs)
    end
    println(Npackets, ", ", total_N)
    times = zeros(total_N);
    x = zeros(total_N, Npackets, 2);
    k = zeros(total_N, Npackets, 2);
    u = zeros(total_N, Npackets, 2);
    base_index = 0
    for idx=indices
        filename = filename_function(idx)
        next_file = nothing
        if idx < indices[end]
            next_file = jldopen(filename_function(idx+1), "r")
        end
    	println("Reading file: " * filename)
        jldopen(filename, "r") do file
    		N = length(keys(file[packet_group * "t"]));
    		index = 1;
    		for ts in keys(file[packet_group * "t"])[1:end-1]
    			times[base_index + index] = file[packet_group * "t/$ts"];
                x[base_index + index, :, :] = file[packet_group * "x/$ts"][packet_idxs, :];
                k[base_index + index, :, :] = file[packet_group * "k/$ts"][packet_idxs, :];
                if(load_velocity)
                    u[base_index + index, :, :] = file[packet_group * "u/$ts"][packet_idxs, :];
                end
    			index += 1;
    		end
            ts = keys(file[packet_group * "t"])[end]
            if haskey(file, packet_group * "u/$ts")
                x[base_index + index, :, :] =      file[packet_group * "x/$ts"][packet_idxs, :]
                k[base_index + index, :, :] =      file[packet_group * "k/$ts"][packet_idxs, :]
                u[base_index + index, :, :] =      file[packet_group * "u/$ts"][packet_idxs, :]
            elseif haskey(file, packet_group * "k/$ts")
                x[base_index + index, :, :] =      file[packet_group * "x/$ts"][packet_idxs, :]
                k[base_index + index, :, :] =      file[packet_group * "k/$ts"][packet_idxs, :]
                u[base_index + index, :, :] = next_file[packet_group * "u/$ts"][packet_idxs, :]
            elseif haskey(file, packet_group * "x/$ts")
                x[base_index + index, :, :] =      file[packet_group * "x/$ts"][packet_idxs, :]
                k[base_index + index, :, :] = next_file[packet_group * "k/$ts"][packet_idxs, :]
                u[base_index + index, :, :] = next_file[packet_group * "u/$ts"][packet_idxs, :]
            else
                x[base_index + index, :, :] = next_file[packet_group * "x/$ts"][packet_idxs, :]
                k[base_index + index, :, :] = next_file[packet_group * "k/$ts"][packet_idxs, :]
                u[base_index + index, :, :] = next_file[packet_group * "u/$ts"][packet_idxs, :]
            end
        	base_index += N
            if !isnothing(next_file)
                close(next_file)
            end
        end
    end
    if(load_velocity)
        return times, x, k, u;
    else
        return times, x, k;
    end
end
