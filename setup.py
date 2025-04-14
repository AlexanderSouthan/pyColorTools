from setuptools import setup, find_packages

setup(
    name='pyColorTools',
    version='0.0.1',
    packages=find_packages(where='src'),
    install_requires=['numpy', 'pandas'],
)
