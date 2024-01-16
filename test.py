import torch
import time
import pandas as pd

# Ensure that PyTorch is using the GPU
device = torch.device("mps")

def benchmark_matrix_multiplication(size, device):
    # Create two square matrices
    a = torch.rand(size, size, device=device)
    b = torch.rand(size, size, device=device)
    
    # Warm-up run
    c = torch.matmul(a, b)

    # Timing
    start_time = time.time()
    for i in range(100):
        c = torch.matmul(a, b)
    end_time = time.time()

    return (end_time - start_time)/100


def benchmark_softmax(size, device):
    # Create a matrix
    a = torch.rand(size, size, device=device)
    
    # Warm-up run
    _ = torch.softmax(a, dim=1)

    # Timing
    start_time = time.time()
    for i in range(10000):
        _ = torch.relu(a)
    end_time = time.time()

    return (end_time - start_time)/100

# Parameters
start_size = 512    # Starting matrix size (128x128)
max_size = 1024     # Maximum matrix size (8192x8192)
growth_factor = 64  # Factor to grow the matrix size at each step

# Record results
results = []
size = start_size

while size <= max_size:
    time_taken = benchmark_softmax(size, device)
    results.append({"Matrix Size": size, "Time (s)": time_taken})
    print(f"Matrix Size: {size}, Time: {time_taken:.6f}s")
    size += growth_factor

# Matrix Size: 512, Time: 0.004124s
# Matrix Size: 576, Time: 0.004021s
# Matrix Size: 640, Time: 0.004037s
# Matrix Size: 704, Time: 0.003937s
# Matrix Size: 768, Time: 0.004079s
# Matrix Size: 832, Time: 0.004180s
# Matrix Size: 896, Time: 0.005871s
# Matrix Size: 960, Time: 0.007299s
# Matrix Size: 1024, Time: 0.008898s