import os
from distutils.core import setup

src_dir = os.path.abspath(os.path.join(__file__, '../src'))

setup(
    name='safety_net',
    version='0.1',
    description="Safety benchmarks for reinforcement learning",
    package_dir={'safety_net': 'src'},
    packages=['safety_net']
)
