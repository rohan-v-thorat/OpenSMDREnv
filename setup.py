from setuptools import find_packages, setup

with open("README.md", "r") as f:
    readme = f.read()

setup(
    name="OpenSMDREnv",
    version="0.0.1",
    author="Rohan Thorat",
    author_email="rohanthorat2@gmail.com",
    description="Spring-Mass-Damper Reinforcement learning Environment",
    long_description=readme,
    url="https://github.com/rohan-v-thorat/OpenSMDREnv/",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    packages=find_packages(exclude=[
        
    ]),
    python_requires=">=3.6",
    install_requires=[
        "numpy", "gymnasium", "scipy"
    ]
)
