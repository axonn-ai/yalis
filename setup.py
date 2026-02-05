from setuptools import setup
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
)
import glob
import os

# Collect all source files for vllm_ops
vllm_cu_sources = glob.glob("yalis/external/csrc/vllm/*.cu") + glob.glob(
    "yalis/external/csrc/vllm/moe/*.cu"
)
vllm_cpp_sources = glob.glob("yalis/external/csrc/vllm/*.cpp") + glob.glob(
    "yalis/external/csrc/vllm/moe/*.cpp"
)
vllm_ops_sources = sorted(vllm_cu_sources + vllm_cpp_sources)
vllm_include_dirs = [
    os.path.abspath("yalis/external/csrc/vllm"),
    os.path.abspath("yalis/external/csrc/vllm/moe"),
]

setup(
    name="yalis",
    version="0.1.0",
    packages=[
        "yalis",
        "yalis.tensor_parallel",
        "yalis.external",
        "yalis.attention",
    ],
    package_dir={"yalis": "yalis"},
    # Other metadata for your package
    ext_modules=[
        CppExtension(
            name="kvcache_manager",
            sources=["yalis/attention/paged_kv_cache.cpp"],
            extra_compile_args=["-O3"],
        ),
        CppExtension(
            name="vllm_ops",
            sources=vllm_ops_sources,
            extra_compile_args=["-O3"],
            include_dirs=vllm_include_dirs,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",  # Ensure PyTorch is installed
        "flask",
    ],
    author="Siddharth Singh, Prajwal Singhania, Lannie Dalton Hough, Ishan Revankar",  # noqa: E501
    description="An easy-to-use library for LLM inference",
)
