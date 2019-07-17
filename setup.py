import os
import glob
import setuptools
import numpy

ext_path = os.path.abspath(os.path.join(__file__, '../safelife/speedups'))
levels_path = os.path.abspath(os.path.join(__file__, '../safelife/levels'))

data_files = ['*.png']
data_files += glob.glob(os.path.join(levels_path, '**/*.npz'), recursive=True)

setuptools.setup(
    name='safety-net',
    version='0.1.dev3',
    author="Carroll L. Wainwright",
    description="Safety benchmarks for reinforcement learning",
    # package_dir={'safelife': src_dir},
    packages=['safelife'],
    package_data={'safelife': data_files},
    install_requires=[
        "pyemd==0.5.1",
        "numpy>=1.11.0",
        "scipy>=1.0.0",
        "gym>=0.12",
        "imageio>=2.5.0",
        "tensorflow>=1.13,<2.0",
    ],
    ext_modules=[
        setuptools.Extension(
            'safelife.speedups',
            define_macros=[
                ('PY_ARRAY_UNIQUE_SYMBOL', 'safelife_speedups'),
                ('NPY_NO_DEPRECATED_API', 'NPY_1_11_API_VERSION')
            ],
            include_dirs=[
                ext_path,
                numpy.get_include()],
            sources=glob.glob(os.path.join(ext_path, '*.c')),
            extra_compile_args=[
                '-O3',
                '-Wno-shorten-64-to-32',
                # '-std=c++11',
                # '-stdlib=libc++',
                # '-mmacosx-version-min=10.9',
                '-Wno-c++11-extensions',
            ]
        ),
    ]
)
