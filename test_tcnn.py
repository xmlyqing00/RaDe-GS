import commentjson as json
import tinycudann as tcnn
import torch

with open("test_tcnn_config_hash.json") as f:
	config = json.load(f)

n_input_dims, n_output_dims = 3, 3

# Option 1: efficient Encoding+Network combo.
model = tcnn.NetworkWithInputEncoding(
	n_input_dims, n_output_dims,
	config["encoding"], config["network"]
)

# Option 2: separate modules. Slower but more flexible.
encoding = tcnn.Encoding(n_input_dims, config["encoding"])
network = tcnn.Network(encoding.n_output_dims, n_output_dims, config["network"])
model = torch.nn.Sequential(encoding, network)

model = model.cuda()
print(model)

input = torch.randn(1, n_input_dims).cuda()
output = model(input)
print(output)