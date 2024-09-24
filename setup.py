from setuptools import setup, find_packages

setup(
    name="manipulator_gym",
    version="0.1.1",
    description="A gym environment for manipulator robots with common abstracted interfaces.",
    author="youliangtan",
    author_email="tan_you_liang@hotmail.com",
    url="https://github.com/rail-berkeley/manipulator_gym",
    packages=find_packages(),
    license="MIT",
    install_requires=[
        "numpy==1.24.3",  # TODO: I changed these, might need to change it back
        "gym >= 0.26",
        "kinpy",
        "pyspacemouse",
        "pynput",
        # Add any other dependencies here
    ],
)
