#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

# Define CUDA architectures to compile for (CUDA 11.8 compatible)
# This covers most modern GPUs from RTX 20/30/40 series, GTX 16/20 series, etc.
cuda_arch_flags = [
    "-arch=sm_50",   # Maxwell (GTX 750, 950, etc.)
    "-arch=sm_60",   # Pascal (GTX 1060, 1070, 1080, RTX 20 series)
    "-arch=sm_70",   # Volta (Tesla V100)
    "-arch=sm_75",   # Turing (RTX 20 series, GTX 16 series)
    "-arch=sm_80",   # Ampere (RTX 30 series, A100)
    "-arch=sm_86",   # Ampere (RTX 30 series mobile)
    "-arch=sm_89",   # Ada Lovelace (RTX 40 series)
]

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={
                "nvcc": cuda_arch_flags, 
                "cxx": cxx_compiler_flags
            })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
