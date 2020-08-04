import os
import glob
import setuptools


class get_numpy_include(object):
    def __str__(self):
        import numpy
        return numpy.get_include()


base_dir = os.path.abspath(os.path.dirname(__file__))
ext_path = os.path.join(base_dir, 'safelife', 'speedups_src')
levels_path = os.path.join(base_dir, 'safelife', 'levels')

data_files = ['*.png']
data_files += glob.glob(os.path.join(levels_path, '**', '*.npz'), recursive=True)
data_files += glob.glob(os.path.join(levels_path, '**', '*.yaml'), recursive=True)

with open(os.path.join(base_dir, "README.md"), "rt", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='safelife',
    version='1.1.1',
    author="Carroll L. Wainwright",
    author_email="carroll@partnershiponai.org",
    description="Safety benchmarks for reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PartnershipOnAI/safelife",
    packages=['safelife'],
    package_data={'safelife': data_files},
    install_requires=[
        "pyemd==0.5.1",
        "numpy>=1.18.0",
        "scipy>=1.0.0",
        "gym>=0.12.5",
        "imageio>=2.5.0",
        "pyglet>=1.3.2,<=1.5.0",
        "pyyaml>=3.12",
    ],
    ext_modules=[
        setuptools.Extension(
            'safelife.speedups',
            define_macros=[
                ('PY_ARRAY_UNIQUE_SYMBOL', 'safelife_speedups'),
                ('NPY_NO_DEPRECATED_API', 'NPY_1_11_API_VERSION')
            ],
            include_dirs=[ext_path, get_numpy_include()],
            sources=glob.glob(os.path.join(ext_path, '*.c')),
            extra_compile_args=[
                '-O3',
                '-Wno-shorten-64-to-32',
                '-Wno-c++11-extensions',
            ]
        ),
    ],
    entry_points={
        'console_scripts': [
            'safelife = safelife.__main__:run',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
