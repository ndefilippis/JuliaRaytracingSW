using JLD2;

function load_packet_analysis_file(filename)
    times = 0;
    x = 0;
    k = 0;
	jldopen(filename, "r") do file
		N = length(keys(file["packets/t"]));
		Npackets = length(keys(file["packets/x"]));

		times = zeros(N, 1);
		x = zeros(N, Npackets, 2);
		k = zeros(N, Npackets, 2);
		index = 1;
		for ts in keys(file["packets/t"])
			times[index] = file["packets/t/$ts"];
			packet_index = 1;
			for packet in keys(file["packets/x"])
				x[index, packet_index, :] = file["packets/x/$packet/$ts"];
				k[index, packet_index, :] = file["packets/k/$packet/$ts"];
				packet_index = packet_index + 1;
			end
			index = index + 1;
		end
	end
	return times, x, k;
end
