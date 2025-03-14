from setuptools import find_packages, setup

setup(
    name="mujoco_sim",
    author="Lester Li",
    version='0.0.1',
    packages=find_packages(),
    description="a mujoco physics simulator for learning neural scene representation",
    include_package_data=True,
)
