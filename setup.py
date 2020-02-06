from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = [x.strip() for x in f]

setup(
    name='discrete_shocklets',
    version='0.0.1',
    install_requires=requirements,
    packages=find_packages(),
    scripts=['scripts/star'],
)
