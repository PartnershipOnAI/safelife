import os
import glob
import setuptools
import numpy

ext_path = os.path.abspath(os.path.join(__file__, '../safelife/ext_mod'))


setuptools.setup(
    name='safety-net',
    version='0.1',
    author="Carroll L. Wainwright",
    description="Safety benchmarks for reinforcement learning",
    # package_dir={'safelife': src_dir},
    packages=['safelife'],
    ext_modules=[
        setuptools.Extension(
            'safelife._ext',
            define_macros=[
                ('PY_ARRAY_UNIQUE_SYMBOL', 'safelife_ext'),
                ('NPY_NO_DEPRECATED_API', 'NPY_1_11_API_VERSION')
            ],
            include_dirs=[
                ext_path,
                numpy.get_include()],
            sources=glob.glob(os.path.join(ext_path, '*.c')),
            extra_compile_args=[
                '-Wno-shorten-64-to-32',
                # '-std=c++11',
                '-stdlib=libc++',
                '-mmacosx-version-min=10.9',
                '-Wno-c++11-extensions',
            ]
        ),
    ]
)
