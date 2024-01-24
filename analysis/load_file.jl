using JLD2;

function load_packet_analysis_file(filename)
    packet_group = "p/"
    times = 0;
    x = 0;
    k = 0;
	jldopen(filename, "r") do file
		N = length(keys(file[packet_group * "t"]));
		Npackets = length(keys(file[packet_group * "x"]));

		times = zeros(N, 1);
		x = zeros(N, Npackets, 2);
		k = zeros(N, Npackets, 2);
		index = 1;
		for ts in keys(file[packet_group * "t"])
			times[index] = file[packet_group * "t/$ts"];
			packet_index = 1;
			for packet in keys(file[packet_group * "x"])
				x[index, packet_index, :] = file[packet_group * "x/$packet/$ts"];
				k[index, packet_index, :] = file[packet_group * "k/$packet/$ts"];
				packet_index = packet_index + 1;
			end
			index = index + 1;
		end
	end
	return times, x, k;
end
