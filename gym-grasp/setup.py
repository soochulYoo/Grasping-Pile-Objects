from setuptools import setup

setup(
    name='gym_grasp',
    version='0.0.1',
    install_requires=['click', 'gym==0.13', 'mujoco-py<2.1,>=2.0', 'termcolor',]
)