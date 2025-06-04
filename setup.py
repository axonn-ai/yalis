from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='yalis',
    version='0.1.0',
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
            name='kvcache_manager',  # the module name you will import in Python
            sources=['yalis/attention/paged_kv_cache.cpp'],  # adjust the path as needed
            extra_compile_args=['-O3'],  # optional, for optimization
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'torch',  # Ensure PyTorch is installed
    ],
    author='Siddharth Singh, Prajwal Singhania, Lannie Dalton Hough, Ishan Revankar',
    description='An easy-to-use library for LLM inference'
)
