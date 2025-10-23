import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

bvh_include_dirs = torch.utils.cpp_extension.include_paths() + ['include/bvh']
bvh_extra_compile_args = {'nvcc': ['-DPRINT_TIMINGS=0',
                                   '-DDEBUG_PRINT=0',
                                   '-DERROR_CHECKING=1',
                                   '-DNUM_THREADS=256',
                                   '-DBVH_PROFILING=0',
                                   ],
                          'cxx': []}

setup(
    name='custom_cuda_utils',
    version='1.5',
    ext_modules=[
        # CUDAExtension('raycast_cuda', [
        #     'raycast_cuda.cpp',
        #     'raycast_cuda_kernel.cu',
        # ]),
        # CUDAExtension('project_features_cuda', [
        #     'project_image_cuda.cpp',
        #     'project_image_cuda_kernel.cu',
        # ]),
        CUDAExtension('custom_cuda_utils', [
            'cuda_utils.cpp',
            'cuda_utils_kernel.cu',
        ]),
        # CUDAExtension('custom_distance_queries', ['bvh.cpp', 'bvh_cuda_op.cu'],
        # include_dirs=bvh_include_dirs,
        # extra_compile_args=bvh_extra_compile_args),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
