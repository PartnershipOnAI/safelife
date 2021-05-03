import os
import glob
import setuptools
import platform
import argparse


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

parser = argparse.ArgumentParser()
parser.add_argument('--py-limited-api')
py_limited_api = parser.parse_known_args()[0].py_limited_api
if py_limited_api:
    # Only set up the limited API macros if we request it.
    # Note that there is a bug in gcc before version 10 which will break
    # compilation when the macro is set.
    assert py_limited_api in ['cp34', 'cp35', 'cp36', 'cp37', 'cp38', 'cp39']
    major, minor = py_limited_api[2:]
    py_limited_api_macro = [
        ('Py_LIMITED_API', '0x0{}0{}0000'.format(major, minor))
    ]
else:
    py_limited_api_macro = []

setuptools.setup(
    name='safelife',
    version='1.2.1',
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
            py_limited_api=bool(py_limited_api),
            define_macros=[
                ('PY_ARRAY_UNIQUE_SYMBOL', 'safelife_speedups'),
                ('NPY_NO_DEPRECATED_API', 'NPY_1_11_API_VERSION'),
                *py_limited_api_macro,
            ],
            include_dirs=[ext_path, get_numpy_include()],
            sources=glob.glob(os.path.join(ext_path, '*.c')),
            extra_compile_args=[
                '-O3',
                '-Wno-shorten-64-to-32',
                '-Wno-c++11-extensions',
                '-Wvla',
            ] if platform.system() != 'Windows' else []
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
