import os
import glob
import setuptools
import numpy

ext_path = os.path.abspath(os.path.join(__file__, '../safelife/speedups_src'))
levels_path = os.path.abspath(os.path.join(__file__, '../safelife/levels'))

data_files = ['*.png']
data_files += glob.glob(os.path.join(levels_path, '**/*.npz'), recursive=True)
data_files += glob.glob(os.path.join(levels_path, '**/*.json'), recursive=True)

requirements = open('requirements.txt').read()
requirements = [line for line in requirements.split('\n') if line]

setuptools.setup(
    name='SafeLife',
    version='0.1b1',
    author="Carroll L. Wainwright",
    description="Safety benchmarks for reinforcement learning",
    packages=['safelife'],
    package_data={'safelife': data_files},
    install_requires=requirements,
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
                '-Wno-c++11-extensions',
            ]
        ),
    ]
)
