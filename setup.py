from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension
import glob
import os

ops_sources = glob.glob("yalis/external/ops/*.cu") + glob.glob("yalis/external/ops/*.cpp");
print(ops_sources)
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
            name="moe_ops",
            sources= ops_sources,
            extra_compile_args=["-O3"],
            include_dirs=[
                "yalis/external/ops",          
            ]
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=[
        "torch",  # Ensure PyTorch is installed
    ],
    author="Siddharth Singh, Prajwal Singhania, Lannie Dalton Hough, Ishan Revankar",  # noqa: E501
    description="An easy-to-use library for LLM inference",
)
