import torch

from src.model.adapter import Adapter

adapter = Adapter()

dummy_input = torch.randn(
    32,
    768
)

output = adapter(dummy_input)

print()

print("Input Shape")
print(dummy_input.shape)

print()

print("Output Shape")
print(output.shape)

print()

print("Same Shape:", dummy_input.shape == output.shape)

print()

total_parameters = sum(
    p.numel()
    for p in adapter.parameters()
)

print("Trainable Parameters")
print(total_parameters)