from setuptools import setup, find_packages

setup(
    name='manipulator_gym',
    version='0.1.0',
    description='A gym environment for manipulator robots with common abstracted interfaces.',
    author='youliangtan',
    author_email='tan_you_liang@hotmail.com',
    url='https://github.com/rail-berkeley/manipulator_gym',
    packages=find_packages(),
    license='MIT',
    install_requires=[
        'numpy',
        'gym>=0.26.0',
        "kinpy"
        # Add any other dependencies here
    ],
)
