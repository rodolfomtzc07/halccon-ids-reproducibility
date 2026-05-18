

import torch
from src.models.halccon import HALCCONMulticlass

def test_model(input_dim: int, num_classes: int = 13, batch_size: int = 4):
    print(f"\nProbando input_dim={input_dim}")
    model = HALCCONMulticlass(input_dim=input_dim, num_classes=num_classes)
    x = torch.randn(batch_size, input_dim)
    y = model(x)
    print(f"length1={model.length1}, length2={model.length2}, length3={model.length3}")
    print(f"input shape:  {x.shape}")
    print(f"output shape: {y.shape}")
    assert y.shape == (batch_size, num_classes)
    print("OK")

test_model(20, 13)
test_model(24, 13)
test_model(30, 13)