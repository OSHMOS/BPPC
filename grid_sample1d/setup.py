from setuptools import setup
import torch.utils.cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Bypass CUDA version mismatch check
torch.utils.cpp_extension._check_cuda_version = lambda *args, **kwargs: None

setup(
    name='grid_sample1d_cuda',
    ext_modules=[
        CUDAExtension(
            'grid_sample1d_cuda', [
                'grid_sample1d_cuda.cpp',
                'grid_sample1d_cuda_kernel.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-ccbin', '/usr/bin/g++']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    })