import os
import setuptools

src_dir = os.path.abspath(os.path.join(__file__, '../safelife'))

setuptools.setup(
    name='safety-net',
    version='0.1',
    author="Carroll L. Wainwright",
    description="Safety benchmarks for reinforcement learning",
    package_dir={'safelife': src_dir},
    packages=['safelife'],
)
