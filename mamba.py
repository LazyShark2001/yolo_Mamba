# import torch
# from mamba_ssm import Mamba
# import time
# batch, length, dim = 2, 64, 64
# x = torch.randn(batch, length, dim).to("cuda")
# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=16,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
# t1 = time.time()
# y = model(x)
# t2 = time.time()
# assert y.shape == x.shape
# print('mamba运行成功')
# take_time_str = '{:.3f} s'.format(t2 - t1)
# print(take_time_str)
# from mamba_ssm import Mamba2
# model = Mamba2(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=64,  # SSM state expansion factor, typically 64 or 128
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
# t1 = time.time()
# y = model(x)
# t2 = time.time()
# assert y.shape == x.shape
# print('mamba2运行成功')
# take_time_str = '{:.3f} s'.format(t2 - t1)
# print(take_time_str)

import torch
import timeit
# print(torch.cuda.is_available())
from mamba_ssm import Mamba, Mamba2

batch, length, dim = 2, 64, 32
x = torch.randn(batch, length, dim).to("cuda")

def try_mamba1(batch, length, dim, x):
    model = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim, # Model dimension d_model
        d_state=16,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to("cuda")
    y = model(x)
    assert y.shape == x.shape

def try_mamba2(batch, length, dim, x):
    model = Mamba2(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim, # Model dimension d_model
        d_state=64,  # SSM state expansion factor, typically 64 or 128
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
    ).to("cuda")
    y = model(x)
    assert y.shape == x.shape

mamba1_time = timeit.timeit('try_mamba1(batch, length, dim, x)', number=10, globals=globals())
print(f"Mamba 1 took {mamba1_time} seconds")

mamba2_time = timeit.timeit('try_mamba2(batch, length, dim, x)', number=10, globals=globals())
print(f"Mamba 2 took {mamba2_time} seconds")
