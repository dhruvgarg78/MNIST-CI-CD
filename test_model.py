from model_utils import SimpleCNN

model = SimpleCNN()
total_params = sum(p.numel() for p in model.parameters())

assert total_params < 100_000, f"Too many parameters: {total_params}"
assert model.fc2.out_features == 10, "Output layer must have 10 classes"

import torch
sample_input = torch.randn(1, 1, 28, 28)
try:
    out = model(sample_input)
    assert out.shape == (1, 10), f"Invalid output shape: {out.shape}"
    print("Model passes shape and parameter tests.")
except Exception as e:
    raise AssertionError(f"Model failed on input: {e}")